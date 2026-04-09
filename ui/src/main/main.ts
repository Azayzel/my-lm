import { app, BrowserWindow, ipcMain, dialog, shell } from "electron";
import * as path from "path";
import * as fs from "fs";
import { spawn } from "child_process";
import { LLMBridge } from "./llmBridge";
import { ImageBridge } from "./imageBridge";
import { HistoryStore } from "./historyStore";
import { ModelManager } from "./modelManager";
import { TrainBridge, TrainConfig } from "./trainBridge";

// ─── Paths ────────────────────────────────────────────────────────────────────
const ROOT = path.resolve(__dirname, "..", "..", ".."); // ui/../.. => repo root
const SCRIPTS_DIR = path.join(ROOT, "scripts");
const MODELS_DIR = path.join(ROOT, "models");
const OUTPUTS_DIR = path.join(ROOT, "outputs");
const LLM_MODEL_DIR = path.join(MODELS_DIR, "qwen3.5-2b");
const IMAGE_MODEL_DIR = path.join(MODELS_DIR, "realvisxl-v4");
const UPSCALER_PATH = path.join(MODELS_DIR, "upscalers", "4x-UltraSharp.pth");

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
const historyStore = new HistoryStore(
  path.join(app.getPath("userData"), "history.json"),
);
const modelManager = new ModelManager(SCRIPTS_DIR, MODELS_DIR);

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
  const mPath = modelPath || LLM_MODEL_DIR;
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

// ─── Media ───────────────────────────────────────────────────────────────────
ipcMain.handle("media:list", async () => {
  if (!fs.existsSync(OUTPUTS_DIR)) return [];
  const files = fs
    .readdirSync(OUTPUTS_DIR)
    .filter((f) => /\.(png|jpg|jpeg|webp)$/i.test(f))
    .map((f) => {
      const full = path.join(OUTPUTS_DIR, f);
      const stat = fs.statSync(full);
      return { name: f, path: full, size: stat.size, mtime: stat.mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime);
  return files;
});

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

// ─── Training ────────────────────────────────────────────────────────────────
ipcMain.handle("train:start", async (_e, config: TrainConfig) => {
  if (trainBridge?.isRunning()) {
    return { ok: false, error: "Training already running" };
  }
  trainBridge = new TrainBridge(SCRIPTS_DIR, PYTHON);
  const result = trainBridge.start(config, (msg) => {
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
      const proc = spawn(PYTHON, [script, basePath, adapterPath, outputPath], {
        stdio: ["ignore", "pipe", "pipe"],
      });
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

// ─── File picker ──────────────────────────────────────────────────────────────
ipcMain.handle("dialog:openDir", async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ["openDirectory"],
  });
  return result.filePaths[0] ?? null;
});
