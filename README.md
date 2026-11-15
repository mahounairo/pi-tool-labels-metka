# pi-tool-labels-metka

Short labels for [pi](https://github.com/earendil-works/pi)'s tool calls, written by [Metka](https://huggingface.co/mahounairo/metka-1.7b-gguf), a small model that runs next to your agent on whatever is left over: a CPU, a laptop, or a little spare VRAM.

A collapsed tool row shows the label instead of the raw call. The label writes itself over the greyed-out command as it streams in, and when the call ends it turns into its past tense or a failure:

```
$ objdump -d game.exe --start-address=0x4a1f30 | head -60
  ->  Disassemble game.exe starting at 0x4a1f30          (while it runs)
  ->  Disassembled game.exe starting at 0x4a1f30         (when it finishes)
  ->  Failed to disassemble game.exe ...                 (when it fails)
```

ctrl+o shows the full calls as usual.

## Install

```
pi install git:github.com/mahounairo/pi-tool-labels-metka
```

Then run `/labels` in pi. It offers:

- **Use a llama-server I already run**: give its address. Any server with llama.cpp's API or the OpenAI completions API (`/v1/completions`) works, on this machine or another one in your network, as long as it passes the prompt through without a chat template (with Ollama, put `TEMPLATE "{{ .Prompt }}"` in the Modelfile).
- **Install a local server for me**: downloads a tested llama.cpp release (on Linux: CPU or Vulkan; on a Mac: Metal on Apple silicon, CPU on Intel) and a Metka model into pi's folder. pi starts the server when it starts and stops it when it exits. On Windows, run a server yourself and use the first option. `METKA_MODEL_URL` overrides where the model is downloaded from (`{file}` in it stands for the file name), for a mirror.
- **Change what the labeller reads**: whether Metka sees the agent's reasoning before each call (see Context below).
- **Status**: whether the server answers.

Which model: Metka 1.7B (1.8 GB) labels better; Metka 0.6B (640 MB) is the one for laptops and older CPUs. On a recent desktop CPU a typical label takes about half a second with the 0.6B and a bit over a second with the 1.7B; on a GPU, a few hundredths of a second with either.

## Context

Metka can read the last 400 characters of the agent's reasoning and text before each call. With them it labels what a command is *for*, not only what it does, which helps most with commands that don't explain themselves (disassembly at an address, hex dumps). This needs a model whose reasoning is visible, such as a local Qwen. With hosted models that hide their reasoning, turn it off: Metka labels just as well from the call alone.

## When Metka isn't sure

Each label comes with the model's confidence in every word. When it drops below a threshold (0.15 by default), the label is cut back to its confident start, or the row shows the call itself as `Run: <call>` (`Ran:` / `Failed:` once it ends). This catches many labels that would have claimed something the command doesn't do.

## Configuration

`/labels` writes `~/.pi/agent/labeller.json`. The other settings there:

| Key | Default | What it does |
|---|---|---|
| `url` | `http://127.0.0.1:11500` | The server, or a list of servers tried in order (one that fails is skipped for 30 s) |
| `context` | `full` | `full` (reasoning and text), `said` (text only) or `none` |
| `minConfidence` | `0.15` | Below this, a label is cut or the command is shown; `0` turns it off |
| `skipTools` | `["read", "edit", "write"]` | Tools left alone (these already show just their file) |
| `minChars` | `0` | Calls shorter than this keep their raw text |
| `backlog` | `8` | How many of a resumed session's newest rows get labels |
| `placeholder` | `false` | Show `...` instead of the call until the first label arrives |
| `parallel` | `2` | How many labels are asked for at once |
| `cutWords` | English words | Where an unsure label may be cut short; `[]` turns cutting off |
| `serverBinary`, `serverModel`, `serverArgs` | | A llama-server that pi starts and stops itself (set by the installer) |
| `serverGpu` | `true` | `false` hides the GPU from a CUDA build that should run on the CPU |

A `url` entry can also be an object for a model behind [llama-swap](https://github.com/mostlygeek/llama-swap): `{ "url": ".../upstream/<model>", "running": ".../running", "notWhile": ["<model>"], "direct": "<the server's own url>" }`. The labeller checks llama-swap's `/running` first: it skips the entry (and goes on to the next) while one of the `notWhile` models is loaded or loading, talks to the server directly once the model is ready, and otherwise goes through llama-swap, which loads it.

Labels are cached in `~/.pi/agent/labeller-cache.jsonl`, so a resumed session doesn't ask again.

## For other extensions

Other extensions can get labels from `globalThis[Symbol.for("pi-labeller")]`: `labelFor(tool, args, state)` asks for one, and `cached(tool, args)` returns a label only if it is already known, without asking the server.

## License

Apache 2.0. The Metka models are Apache 2.0 too; see their model cards for training data and credits.
