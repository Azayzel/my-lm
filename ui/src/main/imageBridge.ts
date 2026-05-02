import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as readline from "readline";

export class ImageBridge {
  private proc: ChildProcess | null = null;
  private _ready = false;
  private _running = false;

  constructor(
    private scriptsDir: string,
    private modelDir: string,
    private upscalerPath: string,
    private outputDir: string,
    private python: string = "python",
    private faceDetectorPath: string = "",
    private nsfwSegModelDir: string = "",
  ) {}

  isRunning() {
    return this._running;
  }
  isReady() {
    return this._ready;
  }

  start(
    onEvent: (msg: object) => void,
  ): Promise<{ ok: boolean; message?: string; error?: string }> {
    return new Promise((resolve) => {
      const bridge = path.join(this.scriptsDir, "image_bridge.py");
      const args = [bridge, this.modelDir, this.upscalerPath, this.outputDir];
      if (this.faceDetectorPath) args.push(this.faceDetectorPath);
      else args.push(""); // placeholder so nsfw_seg_model_dir is at correct index
      if (this.nsfwSegModelDir) args.push(this.nsfwSegModelDir);
      this.proc = spawn(this.python, args, {
        stdio: ["pipe", "pipe", "pipe"],
      });
      this._running = true;

      const rl = readline.createInterface({
        input: this.proc.stdout!,
        // Default crlfDelay avoids splitting on \r\n issues on Windows
        crlfDelay: Infinity,
      });
      rl.on("line", (line) => {
        if (!line) return;
        try {
          const msg = JSON.parse(line) as {
            type: string;
            [key: string]: unknown;
          };
          if (msg.type === "ready") {
            this._ready = true;
            resolve({ ok: true, message: "Image model ready" });
          } else if (msg.type === "error" && !this._ready) {
            resolve({ ok: false, error: String(msg["message"]) });
          }
          onEvent(msg);
        } catch (err) {
          // Don't silently swallow parse errors — surface them so the user
          // can see what's happening. Trim very long lines to avoid spam.
          const preview = line.length > 200 ? line.slice(0, 200) + "…" : line;
          console.error("[image_bridge] JSON parse error:", err, preview);
          onEvent({
            type: "log",
            text: `[parse error] ${preview}`,
          });
        }
      });

      // Forward all non-empty stderr lines so Python tracebacks and warnings
      // aren't lost. Filter out the common deprecation-warning noise only.
      this.proc.stderr!.on("data", (d: Buffer) => {
        const text = d.toString();
        for (const raw of text.split(/\r?\n/)) {
          const trimmed = raw.trim();
          if (!trimmed) continue;
          if (
            trimmed.includes("FutureWarning") ||
            trimmed.includes("warnings.warn") ||
            trimmed.includes("UserWarning")
          ) {
            continue;
          }
          onEvent({ type: "stderr", text: trimmed });
        }
      });

      this.proc.on("exit", (code) => {
        this._running = false;
        this._ready = false;
        onEvent({ type: "exit", code });
      });

      this.proc.on("error", (err) => {
        this._running = false;
        this._ready = false;
        resolve({ ok: false, error: err.message });
      });
    });
  }

  send(request: object) {
    if (this.proc?.stdin?.writable) {
      this.proc.stdin.write(JSON.stringify(request) + "\n");
    }
  }

  kill() {
    this.proc?.kill();
    this._running = false;
    this._ready = false;
  }
}
