// /labels: setting up where labels come from, without editing labeller.json by hand. Either an existing llama-server
// (a URL), or one installed here: a pinned llama.cpp release and a Metka model, both downloaded into the agent folder,
// then started and stopped with pi. The choice applies at once, no reload.
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { execFileSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";

// A llama.cpp release tested with Metka; newer ones come out several times a day, so new installs don't follow them.
const LLAMA = "b12387";
const MODELS: Record<string, { file: string; size: string; url: string }> = {
  "1.7b": { file: "metka-1.7b-Q8_0.gguf", size: "1.8 GB", url: "https://huggingface.co/mahounairo/metka-1.7b-gguf/resolve/main/metka-1.7b-Q8_0.gguf" },
  "0.6b": { file: "metka-0.6b-Q8_0.gguf", size: "640 MB", url: "https://huggingface.co/mahounairo/metka-0.6b-gguf/resolve/main/metka-0.6b-Q8_0.gguf" },
};
const PORT = 11500;

export type SetupApi = {
  agentDir: string;
  config: () => any;
  apply: (changes: Record<string, unknown>) => Promise<void>;
  probe: (url: string) => Promise<{ ok: boolean; model?: string }>;
};

type Build = { label: string; asset: string; gpu: boolean };

function builds(): Build[] {
  const { platform, arch } = process;
  if (platform === "darwin") return [{ label: "Apple GPU (Metal)", asset: `macos-${arch === "arm64" ? "arm64" : "x64"}`, gpu: arch === "arm64" }];
  if (platform !== "linux") return [];
  const a = arch === "arm64" ? "arm64" : "x64";
  return [
    { label: "CPU: works everywhere, about half a second a label on a desktop CPU", asset: `ubuntu-${a}`, gpu: false },
    { label: "GPU via Vulkan: NVIDIA, AMD or Intel with their drivers; a label in a few hundredths of a second", asset: `ubuntu-vulkan-${a}`, gpu: true },
  ];
}

async function download(ctx: ExtensionCommandContext, url: string, to: string, what: string) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || !res.body) throw new Error(`${what}: ${res.status} ${res.statusText} (${url})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const part = `${to}.part`;
  const out = createWriteStream(part);
  let got = 0, shown = 0;
  for await (const chunk of res.body as any) {
    got += chunk.length;
    if (!out.write(chunk)) await new Promise((r) => out.once("drain", r));
    if (Date.now() - shown > 500) {
      shown = Date.now();
      const mb = (n: number) => (n / 1e6).toFixed(0);
      ctx.ui.setStatus("labels", `Downloading ${what}: ${mb(got)}${total ? ` of ${mb(total)}` : ""} MB`);
    }
  }
  await new Promise<void>((resolve, reject) => out.end((e: any) => (e ? reject(e) : resolve())));
  renameSync(part, to);
}

function find(dir: string, name: string): string | undefined {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (e === name && statSync(p).isFile()) return p;
    if (statSync(p).isDirectory()) { const f = find(p, name); if (f) return f; }
  }
}

async function askContext(ctx: ExtensionCommandContext) {
  const full = "Reasoning and text: the most accurate labels; needs a model whose reasoning is visible (local models like Qwen)";
  const off = "Off: the call alone; best for hosted models that hide their reasoning";
  const pick = await ctx.ui.select(
    "What should the labeller read besides the call? Context-enabled labels are more accurate but require a model with reasoning content visible.",
    [full, off],
  );
  return pick === undefined ? undefined : pick === full ? "full" : "none";
}

async function existing(ctx: ExtensionCommandContext, api: SetupApi) {
  const url = (await ctx.ui.input("The llama-server's address", "http://127.0.0.1:8080"))?.trim().replace(/\/+$/, "");
  if (!url) return;
  const found = await api.probe(url);
  if (!found.ok) {
    ctx.ui.notify(`Nothing answered at ${url} (neither /health nor /v1/models). Is the server running, and reachable from here?`, "error");
    return;
  }
  const context = await askContext(ctx);
  if (!context) return;
  await api.apply({ url, context, serverBinary: "", serverModel: "" });
  ctx.ui.notify(`Labels now come from ${url}${found.model ? ` (${found.model})` : ""}.`, "info");
}

async function install(ctx: ExtensionCommandContext, api: SetupApi) {
  const options = builds();
  if (!options.length) {
    ctx.ui.notify(`No ready-made llama.cpp for ${process.platform} here yet; run a llama-server yourself and pick "Use a llama-server I already run".`, "warning");
    return;
  }
  const big = `Metka 1.7B: the better labels (${MODELS["1.7b"].size})`;
  const small = `Metka 0.6B: lighter, for laptops and older CPUs (${MODELS["0.6b"].size})`;
  const which = await ctx.ui.select("Which model?", [big, small]);
  if (!which) return;
  const model = MODELS[which === big ? "1.7b" : "0.6b"];
  const pick = options.length === 1 ? options[0].label : await ctx.ui.select("Run it on", options.map((b) => b.label));
  const chosen = options.find((b) => b.label === pick);
  if (!chosen) return;
  const context = await askContext(ctx);
  if (!context) return;

  const home = join(api.agentDir, "labeller");
  const llamaDir = join(home, `llama-${LLAMA}-${chosen.asset}`);
  const modelPath = join(home, "models", model.file);
  mkdirSync(join(home, "models"), { recursive: true });
  try {
    let binary = existsSync(llamaDir) ? find(llamaDir, "llama-server") : undefined;
    if (!binary) {
      const archive = join(home, `llama-${LLAMA}-${chosen.asset}.tar.gz`);
      await download(ctx, `https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA}/llama-${LLAMA}-bin-${chosen.asset}.tar.gz`, archive, "llama.cpp");
      mkdirSync(llamaDir, { recursive: true });
      execFileSync("tar", ["-xzf", archive, "-C", llamaDir]);
      rmSync(archive, { force: true });
      binary = find(llamaDir, "llama-server");
      if (!binary) throw new Error("llama-server is not in the llama.cpp archive");
    }
    if (!existsSync(modelPath)) await download(ctx, process.env.METKA_MODEL_URL?.replace("{file}", model.file) || model.url, modelPath, model.file);
    ctx.ui.setStatus("labels", "Starting the labeller...");
    await api.apply({
      url: `http://127.0.0.1:${PORT}`, context, serverBinary: binary, serverModel: modelPath,
      serverArgs: ["-ngl", chosen.gpu ? "99" : "0", "-c", "4096", "--parallel", "2"], serverGpu: chosen.gpu,
    });
    const up = await api.probe(`http://127.0.0.1:${PORT}`);
    ctx.ui.notify(up.ok ? `Labeller installed and running (${model.file}, ${chosen.gpu ? "GPU" : "CPU"}).` :
      "Installed, but the server didn't come up; /labels shows its status.", up.ok ? "info" : "warning");
  } catch (e) {
    ctx.ui.notify(`Setup failed: ${e instanceof Error ? e.message : String(e)}`, "error");
  } finally {
    ctx.ui.setStatus("labels", undefined);
  }
}

async function status(ctx: ExtensionCommandContext, api: SetupApi) {
  const c = api.config();
  const urls: string[] = (Array.isArray(c.url) ? c.url : [c.url]).map((u: any) => (typeof u === "string" ? u : u.url));
  const lines = [];
  for (const u of urls) {
    const p = await api.probe(u);
    lines.push(`${p.ok ? "up  " : "down"}  ${u}${p.model ? `  (${p.model})` : ""}`);
  }
  lines.push(`context: ${c.context}`, c.serverBinary ? `started by pi: ${c.serverModel}` : "started by pi: no");
  await ctx.ui.select(`Tool labels\n${lines.join("\n")}`, ["OK"]);
}

export function registerSetup(pi: ExtensionAPI, api: SetupApi) {
  pi.registerCommand("labels", {
    description: "Set up or check where tool-call labels come from",
    handler: async (_args, ctx) => {
      const use = "Use a llama-server I already run";
      const inst = "Install a local server for me";
      const ctxOpt = "Change what the labeller reads (context)";
      const stat = "Status";
      const pick = await ctx.ui.select("Tool labels (Metka)", [use, inst, ctxOpt, stat]);
      if (pick === use) await existing(ctx, api);
      else if (pick === inst) await install(ctx, api);
      else if (pick === ctxOpt) { const context = await askContext(ctx); if (context) await api.apply({ context }); }
      else if (pick === stat) await status(ctx, api);
    },
  });
}
