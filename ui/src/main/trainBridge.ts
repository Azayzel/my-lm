import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as readline from "readline";

export interface TrainConfig {
  model_path: string;
  dataset_path: string;
  output_dir: string;
  epochs: number;
  batch_size: number;
  grad_accum: number;
  lr: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  max_seq_len: number;
  logging_steps: number;
  use_4bit: boolean;
}

export class TrainBridge {
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
    config: TrainConfig,
    onEvent: (msg: object) => void,
  ): { ok: boolean; error?: string } {
    if (this._running) {
      return { ok: false, error: "Training already in progress" };
    }

    const script = path.join(this.scriptsDir, "train_bridge.py");
    try {
      this.proc = spawn(
        this.python,
        [script, "--config", JSON.stringify(config)],
        { stdio: ["ignore", "pipe", "pipe"] },
      );
    } catch (e: unknown) {
      return { ok: false, error: (e as Error).message };
    }

    this._running = true;

    const rl = readline.createInterface({ input: this.proc.stdout! });
    rl.on("line", (line) => {
      try {
        onEvent(JSON.parse(line));
      } catch {
        // non-JSON line — forward as log
        if (line.trim()) onEvent({ type: "log", text: line });
      }
    });

    this.proc.stderr!.on("data", (d: Buffer) => {
      const text = d.toString();
      // stream stderr as log lines (useful for tqdm/warnings)
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
