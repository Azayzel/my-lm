import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as readline from "readline";

export interface BenchmarkConfig {
  models: string;        // comma-separated
  tasks: string;         // comma-separated, "" => all
  conditions: string;    // comma-separated (e.g. "raw,autotune")
  trials: number;
  output?: string;       // absolute path or relative to repo root
  dryRun: boolean;
}

export class BenchmarkBridge {
  private proc: ChildProcess | null = null;
  private _running = false;

  constructor(
    private scriptsDir: string,
    private python: string = "python",
  ) {}

  isRunning() {
    return this._running;
  }

  start(
    config: BenchmarkConfig,
    onEvent: (msg: object) => void,
  ): { ok: boolean; error?: string } {
    if (this._running) {
      return { ok: false, error: "Benchmark already in progress" };
    }

    const script = path.join(this.scriptsDir, "agent_bench.py");
    const args = [script];
    if (config.models) args.push("--models", config.models);
    if (config.tasks) args.push("--tasks", config.tasks);
    if (config.conditions) args.push("--conditions", config.conditions);
    if (Number.isFinite(config.trials) && config.trials > 0) {
      args.push("--trials", String(config.trials));
    }
    if (config.output) args.push("--output", config.output);
    if (config.dryRun) args.push("--dry-run");

    try {
      this.proc = spawn(this.python, args, {
        stdio: ["ignore", "pipe", "pipe"],
        // Force unbuffered Python so the renderer sees progress live.
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });
    } catch (e: unknown) {
      return { ok: false, error: (e as Error).message };
    }

    this._running = true;
    onEvent({ type: "status", message: `Spawned: ${this.python} ${args.join(" ")}` });

    const rl = readline.createInterface({ input: this.proc.stdout! });
    rl.on("line", (line) => {
      if (!line) return;
      onEvent({ type: "log", text: line });
    });

    this.proc.stderr!.on("data", (d: Buffer) => {
      const text = d.toString();
      for (const line of text.split(/\r?\n/)) {
        if (line.trim()) onEvent({ type: "stderr", text: line });
      }
    });

    this.proc.on("exit", (code) => {
      this._running = false;
      onEvent({ type: "exit", code });
    });

    this.proc.on("error", (err) => {
      this._running = false;
      onEvent({ type: "error", message: err.message });
    });

    return { ok: true };
  }

  stop() {
    if (this.proc) {
      try {
        this.proc.kill();
      } catch {
        /* ignore */
      }
    }
    this.proc = null;
    this._running = false;
  }
}
