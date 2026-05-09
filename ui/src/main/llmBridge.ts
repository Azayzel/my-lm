import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as readline from "readline";

export class LLMBridge {
  private proc: ChildProcess | null = null;
  private _ready = false;
  private _running = false;

  constructor(
    private scriptsDir: string,
    private modelPath: string,
    private python: string = "python",
    private bridgeScript: string = "llm_bridge.py",
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
      const bridge = path.join(this.scriptsDir, this.bridgeScript);
      this.proc = spawn(this.python, [bridge, this.modelPath], {
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          PYTHONIOENCODING: "utf-8",
        },
      });
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
            resolve({ ok: true, message: "LLM ready" });
          } else if (msg.type === "error" && !this._ready) {
            resolve({ ok: false, error: String(msg["message"]) });
          }
          onEvent(msg);
        } catch {
          /* ignore parse errors */
        }
      });

      this.proc.stderr!.on("data", (d: Buffer) => {
        const text = d.toString().trim();
        if (text) {
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
