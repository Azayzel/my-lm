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
      this.proc = spawn(
        this.python,
        [bridge, this.modelDir, this.upscalerPath, this.outputDir],
        {
          stdio: ["pipe", "pipe", "pipe"],
        },
      );
      this._running = true;

      const rl = readline.createInterface({ input: this.proc.stdout! });
      rl.on("line", (line) => {
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
        } catch {
          /* ignore */
        }
      });

      this.proc.stderr!.on("data", (d: Buffer) => {
        const text = d.toString();
        if (text.includes("Error") || text.includes("error")) {
          onEvent({ type: "stderr", text });
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
