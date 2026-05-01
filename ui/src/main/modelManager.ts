import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";

export interface ModelInfo {
  name: string;
  path: string;
  type: "llm" | "image" | "upscaler" | "adapter" | "unknown";
  exists: boolean;
  sizeGb?: number;
}

export class ModelManager {
  constructor(
    private scriptsDir: string,
    private modelsDir: string,
  ) {}

  private detectModelType(
    fullPath: string,
    dirName: string,
  ): ModelInfo["type"] {
    if (dirName === "upscalers") return "upscaler";
    if (fs.existsSync(path.join(fullPath, "adapter_config.json")))
      return "adapter";
    if (fs.existsSync(path.join(fullPath, "model_index.json"))) return "image";

    const configPath = path.join(fullPath, "config.json");
    if (fs.existsSync(configPath)) {
      try {
        const cfg = JSON.parse(fs.readFileSync(configPath, "utf-8"));
        if (cfg.model_type || cfg.architectures) return "llm";
        if (cfg._class_name?.includes("Pipeline")) return "image";
      } catch {
        // Ignore malformed config; treat as unknown and continue.
      }
    }

    if (
      fs.existsSync(path.join(fullPath, "adapter_model.safetensors")) ||
      fs.existsSync(path.join(fullPath, "adapter_model.bin"))
    ) {
      return "adapter";
    }

    return "unknown";
  }

  private getDirectorySizeGb(dirPath: string): number {
    let totalBytes = 0;
    const traverse = (dir: string) => {
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            traverse(fullPath);
          } else {
            totalBytes += fs.statSync(fullPath).size;
          }
        }
      } catch {
        // Ignore permission errors
      }
    };
    traverse(dirPath);
    return parseFloat((totalBytes / (1024 * 1024 * 1024)).toFixed(2));
  }

  listModels(): ModelInfo[] {
    if (!fs.existsSync(this.modelsDir)) return [];

    const out: ModelInfo[] = [];

    const walk = (dir: string, depth: number) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        const fullPath = path.join(dir, entry.name);
        const relName = path
          .relative(this.modelsDir, fullPath)
          .replace(/\\/g, "/");
        const type = this.detectModelType(fullPath, entry.name);

        if (type !== "unknown") {
          const sizeGb = this.getDirectorySizeGb(fullPath);
          out.push({
            name: relName,
            path: fullPath,
            type,
            exists: true,
            sizeGb,
          });
          continue;
        }

        // Recurse unknown folders a bit so nested adapters like models/loras/foo appear.
        if (depth < 2) {
          walk(fullPath, depth + 1);
        }
      }
    };

    walk(this.modelsDir, 0);
    out.sort((a, b) => a.name.localeCompare(b.name));
    return out;
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
