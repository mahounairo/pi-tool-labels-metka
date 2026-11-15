// labeller: short labels for tool calls, from a small model (a LoRA on Qwen3 0.6B or 1.7B, served by llama-server
// or any server with the OpenAI completions API).
// A collapsed tool row shows the label instead of the call: the label writes itself over the greyed call as it streams
// in, and what is left of the call goes once the label is done. When the call ends the label turns into its past tense
// ("Restart nginx" -> "Restarted nginx", or "Failed to ..."). Expanded rows (ctrl+o) show the call.
// Other extensions get labels from globalThis[Symbol.for("pi-labeller")]:
// labelFor/labelLine queue model work; cached() is a synchronous cache-only peek that never asks the server.
// Config in ~/.pi/agent/labeller.json:
// - url: the llama-server, or a list of them tried in order (one that fails is skipped for 30 s; see Server below)
// - minChars: calls shorter than this stay as they are (0, the default, labels them all)
// - skipTools: tools left alone (default read, edit and write, which already show just their file)
// - backlog: how many of a resumed session's newest rows get labels (default 8); older ones keep their call unless a
//   label is cached
// - minConfidence: a label whose least likely token is under this probability is not shown; the call is, as
//   "Run: <call>" / "Ran: ..." / "Failed: ..." (default 0.15; 0 turns it off)
// - cutWords: where an unsure label may be cut short (English words, as Metka labels in English; [] turns it off)
// - context: what the model reads besides the call: "full" (the agent's reasoning and text before the call, the last
//   400 characters), "said" (its text only) or "none". Metka's context models label better with it, most with reasoning;
//   an older model without context training wants "none"
// - placeholder: true shows "..." instead of the call while the first label is on its way
// - serverBinary (llama-server) and serverModel (the GGUF): with both set and nothing answering at a local url, pi
//   starts that server itself and it ends with pi; serverArgs replaces the default flags. A CUDA build still uses the
//   GPU for long prompts with -ngl 0 unless it is hidden (CUDA_VISIBLE_DEVICES=), which serverGpu: false does; with
//   it, long calls label far faster for a few hundred MB.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { registerSetup } from "./setup.ts";

const AGENT = process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent");
const CONFIG = join(AGENT, "labeller.json");
const CACHE = join(AGENT, "labeller-cache.jsonl");
const DEFAULTS = {
  url: "http://127.0.0.1:11500", minChars: 0, placeholder: false, parallel: 2, backlog: 8, minConfidence: 0.15,
  skipTools: ["read", "edit", "write"], context: "full" as "full" | "said" | "none",
  cutWords: ["from", "to", "around", "with", "in", "on", "for", "into", "and", "at", "by", "using", "via", "as", "then",
    "while", "after", "before", "under", "inside", "over", "through", "across", "within", "without", "onto", "or"],
  serverBinary: "", serverModel: "", serverArgs: ["-ngl", "0", "-c", "4096", "--parallel", "2"], serverGpu: true,
};
const ARGS_MAX = 3000;
const CONTEXT_MAX = 400;
const RETRY_MS = 30000;

type State = "running" | "finished" | "failed";
type Label = {
  text: string; done: boolean; error: boolean; skipped?: boolean; unsure?: boolean; failedAt?: number;
  watchers: Map<string, () => void>;
};

const config = (() => {
  try { return { ...DEFAULTS, ...JSON.parse(readFileSync(CONFIG, "utf8")) }; } catch { return DEFAULTS; }
})();
// url may be a list, tried in order: a server that fails is skipped for RETRY_MS. An entry can also be an object for a
// model behind llama-swap: { url: ".../upstream/<model>", running: llama-swap's /running, notWhile: [models whose
// loading or loaded state means skip this one without asking], direct: the server's own url, used once llama-swap
// says <model> is ready }.
type Server = string | { url: string; running?: string; notWhile?: string[]; direct?: string };
let servers: Server[] = Array.isArray(config.url) ? config.url : [config.url];
const urlOf = (s: Server) => (typeof s === "string" ? s : s.url);
const downUntil = new Map<string, number>();
const findLocal = () => servers.map(urlOf).find((u) => ["127.0.0.1", "localhost"].includes(new URL(u).hostname));
let localUrl = findLocal();
const runningCache = new Map<string, { at: number; models: { model: string; state: string }[] | undefined }>();

// llama-swap's loaded models, asked at most every 2 s; undefined when it doesn't answer.
async function loaded(url: string) {
  const hit = runningCache.get(url);
  if (hit && Date.now() - hit.at < 2000) return hit.models;
  let models: { model: string; state: string }[] | undefined;
  try { models = (await (await fetch(url, { signal: AbortSignal.timeout(500) })).json()).running; } catch {}
  runningCache.set(url, { at: Date.now(), models });
  return models;
}

// Where to send a label for this server right now, or undefined to skip it.
async function target(s: Server) {
  if (typeof s === "string" || !s.running) return urlOf(s);
  const models = await loaded(s.running);
  if (!models) return s.url;
  if (models.some((m) => s.notWhile?.includes(m.model) && m.state !== "stopped")) return undefined;
  const name = /\/upstream\/([^/]+)/.exec(s.url)?.[1];
  return s.direct && models.some((m) => m.model === name && m.state === "ready") ? s.direct : s.url;
}
const labels = new Map<string, Label>();
const queue: (() => Promise<void>)[] = [];
// Labels for rows that were already over when they showed up, all asked for at once by a resumed session: only the
// newest `backlog` of them are made, the rest keep showing their call.
const backlog: { label: Label; job: () => Promise<void> }[] = [];
let active = 0;
let server: ReturnType<typeof spawn> | undefined;
let starting: Promise<void> | undefined;

// JSON as python's json.dumps writes it (", " and ": "), which is what the model saw in training.
function pyJson(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(pyJson).join(", ")}]`;
  if (v && typeof v === "object") return `{${Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(", ")}}`;
  return JSON.stringify(v) ?? "null";
}

// The agent's words before each call, by tool and arguments, as the context models were trained on them: in a message,
// all its reasoning and all its text up to the call, each cut to its last 400 characters, then joined and cut again.
// Rows from before pi started (a resumed session) get theirs from the session file.
const contexts = new Map<string, string>();
const squash = (t: string) => t.split(/\s+/).filter(Boolean).join(" ");
const tail = (t: string) => squash(t).slice(-CONTEXT_MAX);

function record(message: any) {
  if (message?.role !== "assistant" || !Array.isArray(message.content) || config.context === "none") return;
  const thought: string[] = [], said: string[] = [];
  for (const p of message.content) {
    if (p?.type === "thinking") thought.push(p.thinking ?? "");
    else if (p?.type === "text") said.push(p.text ?? "");
    else if (p?.type === "toolCall" && p.arguments && Object.keys(p.arguments).length) {
      const t = tail(thought.join(" ")), s = tail(said.join(" "));
      const text = config.context === "full" ? tail(`${t} ${s}`) : s;
      contexts.set(`${p.name}\0${pyJson(p.arguments)}`, text);
    }
  }
}

// Rows drawn before the session's contexts are read (right after a start or a /reload) would ask for labels without
// their context: new keys, so cached labels were missed and asked for again. Until then rows keep their call, and are
// redrawn once the contexts are in (or after a few seconds, whatever happens).
let contextsReady = false;
const waiting = new Map<string, () => void>();
function contextsRead() {
  if (contextsReady) return;
  contextsReady = true;
  for (const redraw of waiting.values()) redraw();
  waiting.clear();
}

function contextOf(tool: string, args: unknown) {
  return config.context === "none" ? "" : contexts.get(`${tool}\0${pyJson(args)}`) ?? "";
}

function prompt(tool: string, args: unknown, state: State, running?: string) {
  let json = pyJson(args);
  if (json.length > ARGS_MAX) json = json.slice(0, ARGS_MAX) + " [...]";
  const ctx = contextOf(tool, args);
  const said = ctx ? `<context>${ctx}</context>\n` : "";
  const was = state !== "running" && running ? `<running>${running}</running>\n` : "";
  return `<tool>${tool}</tool>\n<args>${json}</args>\n${said}${was}<state>${state}</state>\n<label>`;
}

// The server's url is part of the key, so switching models doesn't bring back the other model's cached labels. The
// context, when there is one, goes last, so a key without it reads as before.
function keyOf(tool: string, args: unknown, state: State, running?: string) {
  const ctx = contextOf(tool, args);
  return `${servers.map(urlOf).join(",")}\0${tool}\0${pyJson(args)}\0${state}\0${running ?? ""}${ctx ? `\0${ctx}` : ""}`;
}

function loadCache() {
  try {
    for (const line of readFileSync(CACHE, "utf8").split("\n")) {
      if (!line) continue;
      const { key, text, unsure } = JSON.parse(line);
      labels.set(key, { text, done: true, error: false, unsure, watchers: new Map() });
    }
  } catch {}
}

function tell(label: Label) {
  for (const fn of label.watchers.values()) fn();
}

async function healthy() {
  try { return (await (await fetch(`${localUrl}/health`, { signal: AbortSignal.timeout(1000) })).json()).status === "ok"; } catch { return false; }
}

// Starts llama-server when it is configured, the url is on this machine and nothing answers there. It runs under a
// small shell that also watches pi's pid, so the server goes with pi even when pi is killed and can't clean up.
function ensureServer() {
  if (!config.serverBinary || !config.serverModel || !localUrl) return Promise.resolve();
  const url = new URL(localUrl);
  starting ??= (async () => {
    if (await healthy()) return;
    const args = [...config.serverArgs, "-m", config.serverModel, "--host", url.hostname, "--port", url.port || "80"];
    const watch = `"$@" & s=$!; trap 'kill $s 2>/dev/null' EXIT; trap exit TERM; while kill -0 ${process.pid} 2>/dev/null && kill -0 $s 2>/dev/null; do sleep 2; done`;
    const env = config.serverGpu ? process.env : { ...process.env, CUDA_VISIBLE_DEVICES: "" };
    server = spawn("bash", ["-c", watch, "labeller", config.serverBinary, ...args], { stdio: "ignore", env });
    server.on("exit", () => { server = undefined; });
    for (let n = 0; n < 60 && server && !(await healthy()); n++) await new Promise((r) => setTimeout(r, 500));
  })().finally(() => { starting = undefined; });
  return starting;
}

function stopServer() {
  try { server?.kill("SIGTERM"); } catch {}
  server = undefined;
}

// A change from /labels: written into labeller.json (keeping what else is there) and used at once. Labels already
// shown stay; new ones go to the new server (its url is part of the cache key).
async function applyConfig(changes: Record<string, unknown>) {
  let saved: Record<string, unknown> = {};
  try { saved = JSON.parse(readFileSync(CONFIG, "utf8")); } catch {}
  writeFileSync(CONFIG, JSON.stringify({ ...saved, ...changes }, null, 2) + "\n");
  Object.assign(config, changes);
  servers = Array.isArray(config.url) ? config.url : [config.url];
  localUrl = findLocal();
  downUntil.clear();
  if ("serverBinary" in changes || "serverModel" in changes || "url" in changes || "serverArgs" in changes) {
    stopServer();
    await starting?.catch(() => {});
    await ensureServer();
  }
}

// Whether a server could answer, without waking a model behind llama-swap (asking its /health would load it).
async function reachable(s: Server) {
  if (typeof s === "string" || !s.running) return (await probe(urlOf(s))).ok;
  try { return (await fetch(s.running, { signal: AbortSignal.timeout(2000) })).ok; } catch { return false; }
}

// llama-server answers /health; any other server with the OpenAI API (Ollama, LM Studio, vLLM...) /v1/models, where the
// model to ask for is picked too (one named metka, else the first).
async function probe(url: string): Promise<{ ok: boolean; model?: string }> {
  try {
    const health = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2000) });
    if ((await health.json().catch(() => ({}))).status === "ok") {
      let model: string | undefined;
      try { model = String((await (await fetch(`${url}/props`, { signal: AbortSignal.timeout(2000) })).json()).model_path ?? "").split("/").pop() || undefined; } catch {}
      return { ok: true, model };
    }
  } catch {}
  try {
    const ids: string[] = ((await (await fetch(`${url}/v1/models`, { signal: AbortSignal.timeout(2000) })).json()).data ?? []).map((m: any) => m.id);
    const model = ids.find((id) => /metka/i.test(id)) ?? ids[0];
    if (model) modelIds.set(url, model);
    return { ok: ids.length > 0, model };
  } catch { return { ok: false }; }
}

function pump() {
  if (backlog.length) {
    const kept = config.backlog > 0 ? backlog.splice(-config.backlog) : [];
    for (const { label } of backlog) { label.skipped = true; tell(label); }
    backlog.length = 0;
    queue.push(...kept.map((b) => b.job));
  }
  while (active < config.parallel && queue.length) {
    // Newest first: a resumed session asks for every row at once, and the rows at the bottom are the ones in view.
    const job = queue.pop()!;
    active++;
    job().finally(() => { active--; pump(); });
  }
}

// Words that start a new part of a label ("Disassemble d3d9.dll | around the crash address"): an unsure label is cut
// before the last of them, so what's left is a whole phrase, never half of one. They belong to the labels' language,
// English for Metka (cutWords in the config for a model that labels in another; [] turns cutting off). The model's own
// chance of ending the label there would be the language-free signal, but it stays near zero mid-label.
const BOUNDARY = new Set<string>(config.cutWords);

// The confident start of a label that turned unsure at `next` (the token it doubted): up to the last boundary word, or
// all of it when the doubted token is itself one; undefined unless at least two words are left.
function keepBefore(text: string, next: string) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (BOUNDARY.has(next.trim().toLowerCase()) && words.length >= 2) return words.join(" ");
  for (let n = words.length - 1; n >= 2; n--) if (BOUNDARY.has(words[n].toLowerCase())) return words.slice(0, n).join(" ");
  return undefined;
}

// One server's answer, streamed into the label. Throws before the first piece if the server can't answer. Each token
// comes with its probability: once one falls under minConfidence the label is cut there (keepBefore) or, without a
// whole phrase to keep, unsure; the rest isn't waited for
// (on held-out tools, 43% of the 1.7B's least sure fifth were wrong against 7% of its surest; at 0.15 it gives up on
// 13% of unknown tools and 2% of familiar calls).
async function ask(url: string, label: Label, text: string) {
  const stop = new AbortController();
  let stopped = false;
  for await (const { content, sure } of tokens(url, text, stop.signal)) {
    if (sure < config.minConfidence) {
      // On held-out tools the cut keeps the right part: "Disassemble d3d9.dll [around the new function]",
      // "Flush kernel caches [and reboot]"; 28 of 56 unsure labels kept a phrase, the rest fall back.
      const kept = keepBefore(label.text, content);
      if (kept) label.text = kept;
      else label.unsure = true;
      stopped = true;
      break;
    }
    label.text += content;
    tell(label);
  }
  // Leaving the loop early already cancels the body; the abort only makes sure the server stops generating.
  if (stopped) try { stop.abort(); } catch {}
}

// Servers without llama.cpp's own /completion (they answer it with 404) are asked through the OpenAI API's
// /v1/completions instead, with the model id their /v1/models gave. The prompt goes in as it is: such a server must not
// wrap it in a chat template (Ollama: TEMPLATE "{{ .Prompt }}" in the Modelfile).
const openaiUrls = new Set<string>();
const modelIds = new Map<string, string>();

// The label's pieces as they stream in, each with the probability of its least likely token (1 when the server gives
// none). Throws before the first piece if the server can't answer.
async function* tokens(url: string, text: string, signal: AbortSignal): AsyncGenerator<{ content: string; sure: number }> {
  const openai = openaiUrls.has(url);
  const probs = config.minConfidence > 0;
  const body = openai
    ? { model: modelIds.get(url) ?? "metka", prompt: text, max_tokens: 32, temperature: 0, stop: ["</label>"], stream: true,
        ...(probs ? { logprobs: 1 } : {}) }
    : { prompt: text, n_predict: 32, temperature: 0, stop: ["</label>"], cache_prompt: true, stream: true, n_probs: probs ? 1 : 0 };
  const res = await fetch(`${url}${openai ? "/v1/completions" : "/completion"}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  });
  if (!openai && (res.status === 404 || res.status === 405)) {
    await res.body?.cancel();
    openaiUrls.add(url);
    if (!modelIds.has(url)) await probe(url);
    yield* tokens(url, text, signal);
    return;
  }
  if (!res.ok || !res.body) throw new Error(`labeller answered ${res.status}`);
  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of res.body as any) {
    buf += decoder.decode(chunk, { stream: true });
    let at: number;
    while ((at = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, at).trim();
      buf = buf.slice(at + 1);
      if (!line.startsWith("data:")) continue;
      if (line.slice(5).trim() === "[DONE]") return;
      const msg = JSON.parse(line.slice(5));
      if (msg.error) throw new Error(msg.error.message ?? "labeller error");
      const choice = msg.choices?.[0];
      const content: string | undefined = openai ? choice?.text : msg.content;
      if (!content) continue;
      // OpenAI's legacy completions give token_logprobs; llama.cpp gives the chat API's content list there too.
      const lp = choice?.logprobs;
      const logprobs: number[] = openai ? lp?.token_logprobs ?? (lp?.content ?? []).map((t: any) => t.logprob)
        : (msg.completion_probabilities ?? []).map((t: any) => t.logprob ?? 0);
      yield { content, sure: Math.min(...logprobs.map((lp) => Math.exp(lp ?? 0)), 1) };
    }
  }
}

async function stream(key: string, label: Label, text: string) {
  if (starting) await starting;
  for (const s of servers) {
    if ((downUntil.get(urlOf(s)) ?? 0) > Date.now()) continue;
    const url = await target(s);
    if (!url) continue;
    try {
      await ask(url, label, text);
      label.text = label.unsure ? "" : label.text.trim();
      if (!label.text && !label.unsure) throw new Error("empty label");
      label.done = true;
      try { appendFileSync(CACHE, JSON.stringify({ key, text: label.text, ...(label.unsure ? { unsure: true } : {}) }) + "\n"); } catch {}
      return tell(label);
    } catch {
      label.text = "";
      label.unsure = false;
      downUntil.set(urlOf(s), Date.now() + RETRY_MS);
    }
  }
  label.error = true;
  label.failedAt = Date.now();
  // Another pi's server that went away with it: start our own, if we may.
  if (config.serverBinary && !server) ensureServer();
  tell(label);
}

// The label for a call in a state, asked for once and then kept; `watch` is called as it streams in and when done.
export function labelFor(tool: string, args: unknown, state: State, running?: string, watch?: [string, () => void], old = false) {
  const key = keyOf(tool, args, state, running);
  let label = labels.get(key);
  if (label?.error && Date.now() - (label.failedAt ?? 0) > RETRY_MS) label = undefined;
  if (!label) {
    const fresh: Label = { text: "", done: false, error: false, watchers: new Map() };
    labels.set(key, fresh);
    const job = () => stream(key, fresh, prompt(tool, args, state, running));
    if (old) backlog.push({ label: fresh, job });
    else queue.push(job);
    // On the next tick, so a resumed session's rows are all in before the newest are picked.
    setTimeout(pump, 0);
    label = fresh;
  }
  if (watch) label.watchers.set(watch[0], watch[1]);
  return label;
}

// The call as pi shows it collapsed, on one line: the command for bash, key=value pairs for the rest.
function rawOf(tool: string, args: any) {
  if (tool === "bash" && typeof args?.command === "string") return args.command.replace(/\s+/g, " ").trim();
  return Object.entries(args ?? {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ").replace(/\s+/g, " ");
}

// One line that is cut to the width it gets, unlike Text, which would wrap a long call over many lines.
// truncateToWidth puts a full reset before its "...", which also drops the row's background there (pi's tool rows are
// tinted by the box around them), so the cut here only ends the bold and the colour.
class Line {
  constructor(public text: string) {}
  render(width: number) {
    if (visibleWidth(this.text) <= width) return [this.text];
    const cut = truncateToWidth(this.text, Math.max(0, width - 3), "").replaceAll("\x1b[0m", "\x1b[22m\x1b[39m");
    return [`${cut}...`];
  }
  invalidate() {}
}

// The label so far, a cursor, then the rest of what it replaces in grey; just the label once it is done. What it
// replaces stays as it was until the label differs from it: before the first piece, and while the label only repeats
// its start ("Read config.ts" -> "Read config.ts" changes nothing, "Check x" -> "Checked x" starts at the "ed").
function draw(theme: any, label: Label, from: string, style: (s: string) => string) {
  if (label.done) return style(label.text);
  if (from.startsWith(label.text)) return style(from);
  const rest = from.slice(label.text.length + 1);
  return `${style(label.text)}${theme.fg("accent", "_")}${theme.fg("dim", rest)}`;
}

// Rows seen while their call ran. A row that is already over when it first shows up (a resumed session, a job that
// ended before the pane looked) gets its final label written straight over the call; its running label is only asked
// for, since the past tense is written from it, and never shown.
const sawRunning = new Set<string>();

// A call's label in its state as one styled line, written over `from` (the call itself) while it streams in; the
// past tense replaces the running label at once. Undefined when the labeller can't be reached. For other extensions too.
export function labelLine(theme: any, tool: string, args: unknown, state: State, from: string, watch: [string, () => void],
  style: (s: string) => string = (s) => theme.fg("toolTitle", theme.bold(s))) {
  if (!contextsReady && config.context !== "none") { waiting.set(watch[0], watch[1]); return undefined; }
  if (state === "running") sawRunning.add(watch[0]);
  const running = labelFor(tool, args, "running", undefined, watch, !sawRunning.has(watch[0]));
  // Skipped (old rows) leave the call as it was; an unsure label gives way to the call with its state in front.
  const plain = () => style(`${state === "running" ? "Run" : state === "finished" ? "Ran" : "Failed"}: ${from}`);
  if (running.skipped) return undefined;
  if (running.unsure) return plain();
  if (state === "running") return running.error ? undefined : draw(theme, running, config.placeholder ? "..." : from, style);
  if (running.error) return undefined;
  const settled = !sawRunning.has(watch[0]);
  if (!running.done) return settled ? style(from) : draw(theme, running, from, style);
  const final = labelFor(tool, args, state, running.text, watch, settled);
  if (final.skipped) return undefined;
  // The past tense swaps in whole once it's there: typed over the running label it read as the same text typed twice.
  // It falls back to the running label rather than the call when it can't be had.
  if (!settled) return style(final.done && !final.unsure ? final.text : running.text);
  if (final.unsure) return plain();
  return final.error ? undefined : draw(theme, final, from, style);
}

// Left as they are: tools in skipTools (read, edit and write already show just their file), tools without a call
// renderer of their own (pi's fallback also draws their expanded arguments), and tools that draw their own frame
// (renderShell "self", like edit's diff box), which a plain label line would lose.
function wrap(tool: string, base: any) {
  if (!base?.renderCall || base.renderShell === "self" || config.skipTools.includes(tool)) return base;
  return {
    ...base,
    renderCall(args: any, theme: any, ctx: any) {
      const st = ctx.state;
      // The tool's own renderer still runs every time: it keeps its timers and state, and it is what an expanded row,
      // a short call or an unreachable labeller shows.
      const own = base.renderCall(args, theme, { ...ctx, lastComponent: st.labellerOwn });
      st.labellerOwn = own;
      const raw = rawOf(tool, args);
      // Rows replayed from a saved session never hear that their arguments are complete, but they have a result.
      const complete = ctx.argsComplete || ctx.executionStarted || !ctx.isPartial;
      if (ctx.expanded || !complete || raw.length < config.minChars) return own;

      let state: State = ctx.isPartial ? "running" : ctx.isError ? "failed" : "finished";
      // A call that only started a background task is over, but its work isn't: an extension that runs background tasks
      // can report their state under globalThis[Symbol.for("pi-bash-jobs")].state(task), and the row stays "running"
      // until the task ends, checked every 2 s while it runs.
      if (st.labellerTask !== undefined && !ctx.isError) {
        const task = (globalThis as any)[Symbol.for("pi-bash-jobs")]?.state(st.labellerTask);
        if (task) state = task;
        st.labellerInvalidate = ctx.invalidate;
        if (task === "running" && !st.labellerPoll) st.labellerPoll = setInterval(() => st.labellerInvalidate?.(), 2000);
        if (task !== "running" && st.labellerPoll) { clearInterval(st.labellerPoll); st.labellerPoll = undefined; }
      }
      const line = labelLine(theme, tool, args, state, raw, [ctx.toolCallId, ctx.invalidate]);
      if (line === undefined) return own;
      const head = tool === "bash" ? "$ " : `${tool} `;
      return new Line(theme.fg("toolTitle", theme.bold(head)) + line);
    },
    ...(base.renderResult && {
      renderResult(result: any, options: any, theme: any, ctx: any) {
        const task = result?.details?.background ? result.details.task : undefined;
        if (task !== undefined && ctx.state.labellerTask !== task) {
          ctx.state.labellerTask = task;
          queueMicrotask(() => ctx.invalidate());
        }
        return base.renderResult(result, options, theme, ctx);
      },
    }),
  };
}

export default function (pi: ExtensionAPI) {
  loadCache();
  (globalThis as any)[Symbol.for("pi-labeller")] = {
    labelFor, labelLine, minChars: config.minChars,
    // for consumers that must never add load (e.g. decorations): the label if we
    // already have a good one for this exact call, undefined otherwise. Pure Map.get.
    cached: (tool: string, args: unknown): string | undefined => {
      const l = labels.get(keyOf(tool, args, "running"));
      return l?.done && !l.error && !l.unsure ? l.text : undefined;
    },
  };
  pi.registerToolRenderer((tool, next) => wrap(tool, next()));
  registerSetup(pi, { agentDir: AGENT, config: () => config, apply: applyConfig, probe });
  pi.on("session_start", async (_e, ctx) => {
    // Nothing set up yet, or nothing answering: say so once, and how to fix it.
    ensureServer().then(async () => {
      if (!ctx.hasUI) return;
      if (!existsSync(CONFIG)) {
        if (!(await probe(urlOf(servers[0]))).ok) ctx.ui.notify("Tool labels: no labeller set up yet. Run /labels to set one up.", "info");
      } else if (!(await Promise.all(servers.map(reachable))).some(Boolean)) {
        ctx.ui.notify("Tool labels: no labeller server answers; calls show as they are. /labels shows the status.", "warning");
      }
    });
    try { for (const e of ctx.sessionManager.getEntries() as any[]) if (e?.type === "message") record(e.message); } catch {}
    contextsRead();
  });
  setTimeout(contextsRead, 3000).unref?.();
  pi.on("message_update", async (e) => record(e.message));
  pi.on("message_end", async (e) => record(e.message));
  process.once("exit", stopServer);
}
