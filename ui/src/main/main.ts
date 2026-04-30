import {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  shell,
  nativeImage,
} from "electron";
import * as path from "path";
import * as fs from "fs";
import { createHash } from "crypto";
import { spawn } from "child_process";
import { LLMBridge } from "./llmBridge";
import { ImageBridge } from "./imageBridge";
import { HistoryStore } from "./historyStore";
import { ModelManager } from "./modelManager";
import { TrainBridge, TrainConfig } from "./trainBridge";
import { BenchmarkBridge, BenchmarkConfig } from "./benchmarkBridge";
import { BookBridge } from "./bookBridge";
import { PromptStore, SavedPrompt } from "./promptStore";
import { MODEL_CATALOG, filterByVram } from "./modelCatalog";
import { ConfigStore, AppConfig } from "./configStore";

// ─── Paths ────────────────────────────────────────────────────────────────────
const ROOT = path.resolve(__dirname, "..", "..", ".."); // ui/../.. => repo root
const SCRIPTS_DIR = path.join(ROOT, "scripts");
const MODELS_DIR = path.join(ROOT, "models");
const OUTPUTS_DIR = path.join(ROOT, "outputs");
const BENCHMARK_RESULTS_DIR = path.join(ROOT, "benchmark_results");
const LLM_MODEL_DIR = path.join(MODELS_DIR, "qwen3.5-2b");
const IMAGE_MODEL_DIR = path.join(MODELS_DIR, "realvisxl-v4");
const UPSCALER_PATH = path.join(MODELS_DIR, "upscalers", "4x-UltraSharp.pth");
const FACE_DETECTOR_PATH = path.join(
  MODELS_DIR,
  "face_detector",
  "face_yolov8n.pt",
);

// Resolve the Python executable: prefer the project .venv, then fall back to PATH
function resolvePython(): string {
  const candidates = [
    path.join(ROOT, ".venv", "Scripts", "python.exe"), // Windows venv
    path.join(ROOT, ".venv", "bin", "python"), // Unix venv
    path.join(ROOT, "venv", "Scripts", "python.exe"),
    path.join(ROOT, "venv", "bin", "python"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return "python"; // system fallback
}
const PYTHON = resolvePython();

fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

// ─── Singletons ───────────────────────────────────────────────────────────────
let mainWindow: BrowserWindow | null = null;
let llmBridge: LLMBridge | null = null;
let imageBridge: ImageBridge | null = null;
let trainBridge: TrainBridge | null = null;
let benchmarkBridge: BenchmarkBridge | null = null;
let bookBridge: BookBridge | null = null;
const historyStore = new HistoryStore(
  path.join(app.getPath("userData"), "history.json"),
);
const promptStore = new PromptStore(
  path.join(app.getPath("userData"), "prompts.json"),
);
const modelManager = new ModelManager(SCRIPTS_DIR, MODELS_DIR);
const configStore = new ConfigStore(
  path.join(app.getPath("userData"), "config.json"),
);
const THUMB_CACHE_DIR = path.join(app.getPath("userData"), "thumb-cache");

function resolveRepoPath(p?: string): string {
  if (!p) return "";
  return path.isAbsolute(p) ? p : path.join(ROOT, p);
}
fs.mkdirSync(THUMB_CACHE_DIR, { recursive: true });

// Seed config from .env on first run (so existing users don't lose their setup)
if (!configStore.isMongoConfigured()) {
  const envPath = path.join(ROOT, ".env");
  if (fs.existsSync(envPath)) {
    try {
      const envContent = fs.readFileSync(envPath, "utf-8");
      const getVal = (key: string) => {
        const m = envContent.match(new RegExp(`^${key}=(.+)$`, "m"));
        return m ? m[1].trim() : "";
      };
      const uri = getVal("MONGODB_URI");
      if (uri) {
        configStore.set({
          mongoUri: uri,
          mongoDb: getVal("MONGODB_DB") || "bookmind",
          embedModel:
            getVal("BOOKMIND_EMBED_MODEL") ||
            "sentence-transformers/all-MiniLM-L6-v2",
          vectorIndex: getVal("BOOKMIND_VECTOR_INDEX") || "vs_books_embedding",
        });
      }
    } catch {
      /* ignore */
    }
  }
}

function runCommand(
  command: string,
  args: string[],
): Promise<{
  ok: boolean;
  stdout: string;
  stderr: string;
  code: number | null;
}> {
  return new Promise((resolve) => {
    const proc = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d: Buffer) => {
      stdout += d.toString();
    });
    proc.stderr.on("data", (d: Buffer) => {
      stderr += d.toString();
    });
    proc.on("error", (err) => {
      resolve({ ok: false, stdout, stderr: err.message, code: -1 });
    });
    proc.on("exit", (code) => {
      resolve({ ok: code === 0, stdout, stderr, code });
    });
  });
}

// ─── Window ───────────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: "#0d0d0f",
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "hidden",
    frame: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false, // allow loading local file:// images in renderer
    },
    icon: path.join(__dirname, "..", "..", "assets", "icon.png"),
  });

  mainWindow.loadFile(
    path.join(__dirname, "..", "..", "src", "renderer", "index.html"),
  );

  mainWindow.webContents.on("did-finish-load", () => {
    mainWindow?.webContents.send("app:paths", {
      outputs: OUTPUTS_DIR,
      llmModel: LLM_MODEL_DIR,
      imageModel: IMAGE_MODEL_DIR,
      python: PYTHON,
    });
  });

  // Open external links in OS browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
}

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  llmBridge?.kill();
  imageBridge?.kill();
  trainBridge?.stop();
  benchmarkBridge?.stop();
  bookBridge?.kill();
  if (process.platform !== "darwin") app.quit();
});

// ─── Window controls ──────────────────────────────────────────────────────────
ipcMain.on("window:minimize", () => mainWindow?.minimize());
ipcMain.on("window:maximize", () => {
  if (mainWindow?.isMaximized()) mainWindow.unmaximize();
  else mainWindow?.maximize();
});
ipcMain.on("window:close", () => mainWindow?.close());

// ─── LLM ──────────────────────────────────────────────────────────────────────
ipcMain.handle("llm:start", async (_e, modelPath: string) => {
  if (llmBridge?.isRunning()) return { ok: true, message: "Already running" };
  const mPath = modelPath ? resolveRepoPath(modelPath) : LLM_MODEL_DIR;
  llmBridge = new LLMBridge(SCRIPTS_DIR, mPath, PYTHON);
  return llmBridge.start((msg) => {
    mainWindow?.webContents.send("llm:event", msg);
  });
});

ipcMain.handle("llm:chat", async (_e, request: object) => {
  if (!llmBridge?.isReady()) {
    return { ok: false, error: "LLM not ready. Please start the model first." };
  }
  llmBridge.send(request);
  return { ok: true };
});

ipcMain.handle("llm:stop", async () => {
  llmBridge?.kill();
  llmBridge = null;
  return { ok: true };
});

ipcMain.handle("llm:status", async () => ({
  running: llmBridge?.isRunning() ?? false,
  ready: llmBridge?.isReady() ?? false,
}));

// ─── Image Generation ─────────────────────────────────────────────────────────
ipcMain.handle("image:start", async (_e, modelPath?: string) => {
  if (imageBridge?.isRunning()) return { ok: true, message: "Already running" };
  const mPath = modelPath || IMAGE_MODEL_DIR;
  imageBridge = new ImageBridge(
    SCRIPTS_DIR,
    mPath,
    UPSCALER_PATH,
    OUTPUTS_DIR,
    PYTHON,
    FACE_DETECTOR_PATH,
  );
  return imageBridge.start((msg) => {
    mainWindow?.webContents.send("image:event", msg);
  });
});

ipcMain.handle("image:generate", async (_e, request: object) => {
  if (!imageBridge?.isReady()) {
    return {
      ok: false,
      error: "Image model not ready. Please start it first.",
    };
  }
  imageBridge.send(request);
  return { ok: true };
});

ipcMain.handle("image:stop", async () => {
  imageBridge?.kill();
  imageBridge = null;
  return { ok: true };
});

ipcMain.handle("image:status", async () => ({
  running: imageBridge?.isRunning() ?? false,
  ready: imageBridge?.isReady() ?? false,
}));

// ─── History ─────────────────────────────────────────────────────────────────
ipcMain.handle("history:get", async () => historyStore.getAll());
// eslint-disable-next-line @typescript-eslint/no-explicit-any
ipcMain.handle("history:save", async (_e: any, entry: any) =>
  historyStore.save(entry),
);
ipcMain.handle("history:clear", async () => historyStore.clear());
// eslint-disable-next-line @typescript-eslint/no-explicit-any
ipcMain.handle("history:delete", async (_e: any, id: string) =>
  historyStore.delete(id),
);

// ─── Saved Prompts ───────────────────────────────────────────────────────────
ipcMain.handle("prompts:list", async () => promptStore.getAll());
ipcMain.handle(
  "prompts:save",
  async (_e, entry: Omit<SavedPrompt, "id" | "timestamp">) =>
    promptStore.save(entry),
);
ipcMain.handle(
  "prompts:update",
  async (_e, id: string, patch: Partial<SavedPrompt>) =>
    promptStore.update(id, patch),
);
ipcMain.handle("prompts:delete", async (_e, id: string) =>
  promptStore.delete(id),
);

// ─── Media ───────────────────────────────────────────────────────────────────
// ─── Media (with folder support) ─────────────────────────────────────────
ipcMain.handle("media:list", async (_e, subdir?: string) => {
  const dir = subdir ? path.join(OUTPUTS_DIR, subdir) : OUTPUTS_DIR;
  if (!fs.existsSync(dir)) return { folders: [], files: [] };

  const entries = await fs.promises.readdir(dir, { withFileTypes: true });

  const folders = entries
    .filter((e) => e.isDirectory())
    .map((e) => ({
      name: e.name,
      path: path.join(dir, e.name),
      rel: subdir ? `${subdir}/${e.name}` : e.name,
    }))
    .sort((a, b) => a.name.localeCompare(b.name));

  const files = entries
    .filter((e) => e.isFile() && /\.(png|jpg|jpeg|webp)$/i.test(e.name))
    .map((e) => ({
      name: e.name,
      path: path.join(dir, e.name),
      size: 0,
      mtime: 0,
    }))
    // Most generated files include timestamps in the filename, so this is fast and stable.
    .sort((a, b) => b.name.localeCompare(a.name, undefined, { numeric: true }));

  return { folders, files };
});

ipcMain.handle(
  "media:getThumbnail",
  async (_e, filePath: string, maxSize = 360) => {
    try {
      const resolved = path.resolve(filePath);
      const outputsResolved = path.resolve(OUTPUTS_DIR);
      if (!resolved.startsWith(outputsResolved)) {
        return { ok: false, error: "File is outside outputs directory" };
      }
      if (!fs.existsSync(resolved)) {
        return { ok: false, error: "File not found" };
      }

      const stat = await fs.promises.stat(resolved);
      const key = createHash("sha1")
        .update(`${resolved}|${stat.mtimeMs}|${maxSize}`)
        .digest("hex");
      const cachedPath = path.join(THUMB_CACHE_DIR, `${key}.png`);

      if (fs.existsSync(cachedPath)) {
        return { ok: true, path: cachedPath, cached: true };
      }

      const img = nativeImage.createFromPath(resolved);
      if (img.isEmpty()) {
        return { ok: false, error: "Could not decode image" };
      }

      const sourceSize = img.getSize();
      const maxDim = Math.max(sourceSize.width, sourceSize.height);
      const scale = maxDim > maxSize ? maxSize / maxDim : 1;
      const targetW = Math.max(1, Math.round(sourceSize.width * scale));
      const targetH = Math.max(1, Math.round(sourceSize.height * scale));
      const thumb = img.resize({
        width: targetW,
        height: targetH,
        quality: "good",
      });
      await fs.promises.writeFile(cachedPath, thumb.toPNG());

      return { ok: true, path: cachedPath, cached: false };
    } catch (e: any) {
      return { ok: false, error: e.message };
    }
  },
);

ipcMain.handle("media:createFolder", async (_e, name: string) => {
  try {
    const sanitized = name.replace(/[<>:"/\\|?*]/g, "_").trim();
    if (!sanitized) return { ok: false, error: "Invalid folder name" };
    const full = path.join(OUTPUTS_DIR, sanitized);
    fs.mkdirSync(full, { recursive: true });
    return { ok: true, path: full };
  } catch (e: any) {
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("media:deleteFolder", async (_e, relPath: string) => {
  try {
    const full = path.join(OUTPUTS_DIR, relPath);
    if (!fs.existsSync(full)) return { ok: false, error: "Folder not found" };
    fs.rmSync(full, { recursive: true, force: true });
    return { ok: true };
  } catch (e: any) {
    return { ok: false, error: e.message };
  }
});

ipcMain.handle(
  "media:move",
  async (_e, filePath: string, destFolder: string) => {
    try {
      const destDir = destFolder
        ? path.join(OUTPUTS_DIR, destFolder)
        : OUTPUTS_DIR;
      fs.mkdirSync(destDir, { recursive: true });
      const basename = path.basename(filePath);
      const dest = path.join(destDir, basename);
      fs.renameSync(filePath, dest);
      return { ok: true, newPath: dest };
    } catch (e: any) {
      return { ok: false, error: e.message };
    }
  },
);

ipcMain.handle("media:delete", async (_e, filePath: string) => {
  try {
    fs.unlinkSync(filePath);
    return { ok: true };
  } catch (e: any) {
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("media:open", async (_e, filePath: string) => {
  shell.openPath(filePath);
});

ipcMain.handle("media:show", async (_e, filePath: string) => {
  shell.showItemInFolder(filePath);
});

// ─── BookMind (RAG book recommender) ─────────────────────────────────────────
ipcMain.handle("books:start", async () => {
  if (bookBridge?.isRunning()) return { ok: true, message: "Already running" };
  if (!configStore.isMongoConfigured()) {
    return {
      ok: false,
      error:
        "MongoDB not configured. Go to Settings and set up a connection first.",
    };
  }
  bookBridge = new BookBridge(SCRIPTS_DIR, PYTHON, configStore.toEnv());
  return bookBridge.start((msg) => {
    mainWindow?.webContents.send("books:event", msg);
  });
});

ipcMain.handle("books:query", async (_e, request: object) => {
  if (!bookBridge?.isReady()) {
    return {
      ok: false,
      error: "BookMind bridge not ready. Please start it first.",
    };
  }
  bookBridge.send(request);
  return { ok: true };
});

ipcMain.handle("books:stop", async () => {
  bookBridge?.kill();
  bookBridge = null;
  return { ok: true };
});

ipcMain.handle("books:status", async () => ({
  running: bookBridge?.isRunning() ?? false,
  ready: bookBridge?.isReady() ?? false,
}));

// ─── Training ────────────────────────────────────────────────────────────────
ipcMain.handle("train:start", async (_e, config: TrainConfig) => {
  if (trainBridge?.isRunning()) {
    return { ok: false, error: "Training already running" };
  }
  const resolvedConfig: TrainConfig = {
    ...config,
    model_path: resolveRepoPath(config.model_path),
    dataset_path: resolveRepoPath(config.dataset_path),
    output_dir: resolveRepoPath(config.output_dir),
  };
  trainBridge = new TrainBridge(SCRIPTS_DIR, PYTHON);
  const result = trainBridge.start(resolvedConfig, (msg) => {
    mainWindow?.webContents.send("train:event", msg);
  });
  return result;
});

ipcMain.handle("train:stop", async () => {
  trainBridge?.stop();
  return { ok: true };
});

ipcMain.handle("train:status", async () => ({
  running: trainBridge?.isRunning() ?? false,
}));

// ─── Merge LoRA adapter into base model ──────────────────────────────────────
ipcMain.handle(
  "train:merge",
  async (_e, basePath: string, adapterPath: string, outputPath: string) => {
    return new Promise<{ ok: boolean; error?: string }>((resolve) => {
      const script = path.join(SCRIPTS_DIR, "merge_lora.py");
      const proc = spawn(
        PYTHON,
        [
          script,
          resolveRepoPath(basePath),
          resolveRepoPath(adapterPath),
          resolveRepoPath(outputPath),
        ],
        {
          stdio: ["ignore", "pipe", "pipe"],
        },
      );
      let stderr = "";
      proc.stdout.on("data", (d: Buffer) => {
        mainWindow?.webContents.send("train:event", {
          type: "log",
          text: d.toString(),
        });
      });
      proc.stderr.on("data", (d: Buffer) => {
        stderr += d.toString();
      });
      proc.on("exit", (code) => {
        if (code === 0) resolve({ ok: true });
        else resolve({ ok: false, error: stderr || `exit code ${code}` });
      });
      proc.on("error", (err) => resolve({ ok: false, error: err.message }));
    });
  },
);

// ─── Benchmarking (agent_bench.py) ───────────────────────────────────────────
ipcMain.handle("bench:start", async (_e, config: BenchmarkConfig) => {
  if (benchmarkBridge?.isRunning()) {
    return { ok: false, error: "Benchmark already running" };
  }
  benchmarkBridge = new BenchmarkBridge(SCRIPTS_DIR, PYTHON);
  return benchmarkBridge.start(config, (msg) => {
    mainWindow?.webContents.send("bench:event", msg);
  });
});

ipcMain.handle("bench:stop", async () => {
  benchmarkBridge?.stop();
  return { ok: true };
});

ipcMain.handle("bench:status", async () => ({
  running: benchmarkBridge?.isRunning() ?? false,
  resultsDir: BENCHMARK_RESULTS_DIR,
}));

ipcMain.handle("bench:listResults", async () => {
  try {
    if (!fs.existsSync(BENCHMARK_RESULTS_DIR)) {
      return { ok: true, dir: BENCHMARK_RESULTS_DIR, files: [] };
    }
    const entries = fs.readdirSync(BENCHMARK_RESULTS_DIR, {
      withFileTypes: true,
    });
    const files = entries
      .filter((e) => e.isFile() && e.name.toLowerCase().endsWith(".json"))
      .map((e) => {
        const full = path.join(BENCHMARK_RESULTS_DIR, e.name);
        const stat = fs.statSync(full);
        return {
          name: e.name,
          path: full,
          size: stat.size,
          mtime: stat.mtimeMs,
        };
      })
      .sort((a, b) => b.mtime - a.mtime);
    return { ok: true, dir: BENCHMARK_RESULTS_DIR, files };
  } catch (e) {
    return {
      ok: false,
      dir: BENCHMARK_RESULTS_DIR,
      files: [],
      error: (e as Error).message,
    };
  }
});

ipcMain.handle("bench:getResult", async (_e, filePath: string) => {
  try {
    // Restrict reads to the benchmark results directory.
    const full = path.resolve(filePath);
    const dir = path.resolve(BENCHMARK_RESULTS_DIR);
    if (!full.startsWith(dir + path.sep) && full !== dir) {
      return { ok: false, error: "Path outside benchmark results directory" };
    }
    if (!fs.existsSync(full)) return { ok: false, error: "File not found" };
    const text = fs.readFileSync(full, "utf-8");
    const data = JSON.parse(text);
    return { ok: true, data };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle("bench:openResultsDir", async () => {
  fs.mkdirSync(BENCHMARK_RESULTS_DIR, { recursive: true });
  shell.openPath(BENCHMARK_RESULTS_DIR);
  return { ok: true };
});

// ─── Dataset file picker ─────────────────────────────────────────────────────
ipcMain.handle(
  "dialog:openFile",
  async (_e, filters?: { name: string; extensions: string[] }[]) => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      properties: ["openFile"],
      filters: filters ?? [{ name: "All Files", extensions: ["*"] }],
    });
    return result.filePaths[0] ?? null;
  },
);

// ─── Model management ─────────────────────────────────────────────────────────
ipcMain.handle("models:list", async () => modelManager.listModels());

ipcMain.handle(
  "models:download",
  async (_e, repoId: string, targetDir: string, type: string) => {
    modelManager.download(repoId, targetDir, type, PYTHON, (msg) => {
      mainWindow?.webContents.send("models:download:event", msg);
    });
    return { ok: true };
  },
);

ipcMain.handle("models:exists", async (_e, modelPath: string) =>
  fs.existsSync(modelPath),
);

// ─── Model catalog (curated, filterable by VRAM) ─────────────────────────
ipcMain.handle("catalog:list", async (_e, vramGb?: number) => {
  const installedDirs = new Set(
    modelManager.listModels().map((m) => m.name.toLowerCase()),
  );
  const filtered = vramGb ? filterByVram(vramGb) : MODEL_CATALOG;
  return filtered.map((entry) => ({
    ...entry,
    installed: installedDirs.has(entry.targetDir.toLowerCase()),
  }));
});

// ─── System diagnostics ──────────────────────────────────────────────────────
ipcMain.handle("system:diagnostics", async () => {
  const pyScript = [
    "import json",
    "import importlib.metadata as m",
    "result={'python_executable': __import__('sys').executable, 'python_version': __import__('sys').version.split()[0]}",
    "try:",
    "    result['transformers_version']=m.version('transformers')",
    "except Exception:",
    "    result['transformers_version']=None",
    "print(json.dumps(result))",
  ].join("\n");

  const pyResult = await runCommand(PYTHON, ["-c", pyScript]);
  let parsed: {
    python_executable: string;
    python_version: string;
    transformers_version: string | null;
  } | null = null;

  if (pyResult.ok) {
    try {
      parsed = JSON.parse(pyResult.stdout.trim());
    } catch {
      parsed = null;
    }
  }

  return {
    ok: Boolean(parsed),
    pythonConfigured: PYTHON,
    pythonResolved: parsed?.python_executable ?? null,
    pythonVersion: parsed?.python_version ?? null,
    transformersVersion: parsed?.transformers_version ?? null,
    stderr: pyResult.stderr || null,
    raw: pyResult.stdout || null,
  };
});

ipcMain.handle("system:gpuInfo", async () => {
  const pyGpuScript = [
    "import json",
    "out={'torch_available': False, 'cuda_available': False, 'cuda_version': None, 'device_count': 0, 'devices': []}",
    "try:",
    "    import torch",
    "    out['torch_available'] = True",
    "    out['cuda_available'] = bool(torch.cuda.is_available())",
    "    out['cuda_version'] = torch.version.cuda",
    "    if out['cuda_available']:",
    "        out['device_count'] = torch.cuda.device_count()",
    "        for i in range(out['device_count']):",
    "            props = torch.cuda.get_device_properties(i)",
    "            out['devices'].append({'index': i, 'name': props.name, 'total_memory_gb': round(props.total_memory/(1024**3), 2), 'major': props.major, 'minor': props.minor})",
    "except Exception as e:",
    "    out['error']=str(e)",
    "print(json.dumps(out))",
  ].join("\n");

  const pyResult = await runCommand(PYTHON, ["-c", pyGpuScript]);
  let pyParsed: any = null;
  if (pyResult.ok) {
    try {
      pyParsed = JSON.parse(pyResult.stdout.trim());
    } catch {
      pyParsed = null;
    }
  }

  const smi = await runCommand("nvidia-smi", [
    "--query-gpu=name,driver_version,memory.total,temperature.gpu,utilization.gpu",
    "--format=csv,noheader,nounits",
  ]);

  const nvidiaRows = smi.ok
    ? smi.stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const [name, driver, memoryMb, tempC, util] = line
            .split(",")
            .map((part) => part.trim());
          return {
            name,
            driverVersion: driver,
            memoryMb: Number(memoryMb),
            temperatureC: Number(tempC),
            utilizationPercent: Number(util),
          };
        })
    : [];

  return {
    ok: true,
    python: pyParsed,
    nvidiaSmiAvailable: smi.ok,
    nvidia: nvidiaRows,
    nvidiaError: smi.ok ? null : smi.stderr || "nvidia-smi not available",
  };
});

ipcMain.handle("system:clearThumbnailCache", async () => {
  try {
    await fs.promises.mkdir(THUMB_CACHE_DIR, { recursive: true });
    const names = await fs.promises.readdir(THUMB_CACHE_DIR);
    let removedFiles = 0;
    let removedBytes = 0;

    for (const name of names) {
      const full = path.join(THUMB_CACHE_DIR, name);
      try {
        const stat = await fs.promises.stat(full);
        if (stat.isFile()) {
          removedBytes += stat.size;
          await fs.promises.unlink(full);
          removedFiles += 1;
        }
      } catch {
        // Ignore files that disappear during cleanup.
      }
    }

    return { ok: true, removedFiles, removedBytes };
  } catch (e: any) {
    return { ok: false, error: e.message };
  }
});

// ─── Config / Settings ───────────────────────────────────────────────────────
ipcMain.handle("config:get", async () => configStore.get());

ipcMain.handle("config:set", async (_e, patch: Partial<AppConfig>) =>
  configStore.set(patch),
);

ipcMain.handle("config:testMongo", async (_e, uri: string, dbName: string) => {
  const script = [
    "import sys, json",
    `uri = ${JSON.stringify(uri)}`,
    `db_name = ${JSON.stringify(dbName)}`,
    "try:",
    "    from pymongo import MongoClient",
    "    c = MongoClient(uri, serverSelectionTimeoutMS=8000)",
    "    c.admin.command('ping')",
    "    db = c[db_name]",
    "    colls = db.list_collection_names()",
    "    counts = {n: db[n].count_documents({}) for n in colls}",
    "    print(json.dumps({'ok': True, 'collections': counts}))",
    "except Exception as e:",
    "    print(json.dumps({'ok': False, 'error': str(e)}))",
  ].join("\n");

  return new Promise<{
    ok: boolean;
    collections?: Record<string, number>;
    error?: string;
  }>((resolve) => {
    const proc = spawn(PYTHON, ["-c", script], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    proc.stdout.on("data", (d: Buffer) => {
      stdout += d.toString();
    });
    proc.on("exit", () => {
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        resolve({ ok: false, error: stdout || "Unknown error" });
      }
    });
    proc.on("error", (err) => resolve({ ok: false, error: err.message }));
  });
});

// ─── File picker ──────────────────────────────────────────────────────────────
ipcMain.handle("dialog:openDir", async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ["openDirectory"],
  });
  return result.filePaths[0] ?? null;
});
