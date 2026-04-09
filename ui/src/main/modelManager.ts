import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";

export interface ModelInfo {
  name: string;
  path: string;
  type: "llm" | "image" | "upscaler" | "unknown";
  exists: boolean;
}

export class ModelManager {
  constructor(
    private scriptsDir: string,
    private modelsDir: string,
  ) {}

  listModels(): ModelInfo[] {
    const dirs = fs.existsSync(this.modelsDir)
      ? fs
          .readdirSync(this.modelsDir, { withFileTypes: true })
          .filter((d) => d.isDirectory())
      : [];

    return dirs.map((d) => {
      const fullPath = path.join(this.modelsDir, d.name);
      let type: ModelInfo["type"] = "unknown";
      if (fs.existsSync(path.join(fullPath, "config.json"))) {
        const cfg = JSON.parse(
          fs.readFileSync(path.join(fullPath, "config.json"), "utf-8"),
        );
        if (cfg.model_type || cfg.architectures) type = "llm";
        else if (cfg._class_name?.includes("Pipeline")) type = "image";
      }
      if (fs.existsSync(path.join(fullPath, "model_index.json")))
        type = "image";
      if (d.name === "upscalers") type = "upscaler";
      return { name: d.name, path: fullPath, type, exists: true };
    });
  }

  download(
    repoId: string,
    targetDir: string,
    modelType: string,
    python: string = "python",
    onEvent: (msg: object) => void,
  ) {
    const script = path.join(this.scriptsDir, "model_download.py");
    const proc = spawn(
      python,
      [script, repoId, targetDir, "--type", modelType],
      {
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    const rl = readline.createInterface({ input: proc.stdout });
    rl.on("line", (line) => {
      try {
        onEvent(JSON.parse(line));
      } catch {
        /* ignore */
      }
    });

    proc.stderr.on("data", (d: Buffer) => {
      onEvent({ type: "stderr", text: d.toString() });
    });

    proc.on("exit", (code) => {
      onEvent({ type: "download:exit", code });
    });
  }
}
