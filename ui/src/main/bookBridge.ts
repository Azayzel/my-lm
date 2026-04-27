import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as readline from "readline";

export class BookBridge {
  private proc: ChildProcess | null = null;
  private _ready = false;
  private _running = false;

  constructor(
    private scriptsDir: string,
    private python: string = "python",
    private extraEnv: Record<string, string> = {},
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
      const bridge = path.join(this.scriptsDir, "book_bridge.py");
      try {
        this.proc = spawn(this.python, [bridge], {
          stdio: ["pipe", "pipe", "pipe"],
          env: { ...process.env, ...this.extraEnv },
        });
      } catch (e: unknown) {
        resolve({ ok: false, error: (e as Error).message });
        return;
      }

      this._running = true;

      const rl = readline.createInterface({ input: this.proc.stdout! });
      rl.on("line", (line) => {
        try {
          const msg = JSON.parse(line);
          onEvent(msg);
          // First status line from book_bridge means ready-for-requests
          if (msg.type === "status" && !this._ready) {
            this._ready = true;
            resolve({ ok: true, message: "BookMind bridge ready" });
          }
        } catch {
          /* ignore non-JSON */
        }
      });

      this.proc.stderr!.on("data", (d: Buffer) => {
        const text = d.toString();
        if (/error|traceback/i.test(text)) {
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
