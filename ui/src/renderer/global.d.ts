// Global type declarations for the Lavely renderer

interface LavelyAPI {
  window: {
    minimize(): void;
    maximize(): void;
    close(): void;
  };
  llm: {
    start(
      modelPath?: string,
    ): Promise<{ ok: boolean; message?: string; error?: string }>;
    chat(request: object): Promise<{ ok: boolean; error?: string }>;
    stop(): Promise<{ ok: boolean }>;
    status(): Promise<{ running: boolean; ready: boolean }>;
    onEvent(cb: (msg: BridgeMsg) => void): () => void;
  };
  image: {
    start(
      modelPath?: string,
    ): Promise<{ ok: boolean; message?: string; error?: string }>;
    generate(request: object): Promise<{ ok: boolean; error?: string }>;
    stop(): Promise<{ ok: boolean }>;
    status(): Promise<{ running: boolean; ready: boolean }>;
    onEvent(cb: (msg: BridgeMsg) => void): () => void;
  };
  history: {
    get(): Promise<HistoryEntry[]>;
    save(entry: object): Promise<HistoryEntry>;
    clear(): Promise<void>;
    delete(id: string): Promise<boolean>;
  };
  media: {
    list(): Promise<MediaFile[]>;
    delete(path: string): Promise<{ ok: boolean; error?: string }>;
    open(path: string): Promise<void>;
    show(path: string): Promise<void>;
  };
  models: {
    list(): Promise<ModelInfo[]>;
    download(
      repoId: string,
      targetDir: string,
      type: string,
    ): Promise<{ ok: boolean }>;
    exists(path: string): Promise<boolean>;
    onDownloadEvent(cb: (msg: BridgeMsg) => void): () => void;
  };
  train: {
    start(config: TrainConfig): Promise<{ ok: boolean; error?: string }>;
    stop(): Promise<{ ok: boolean }>;
    status(): Promise<{ running: boolean }>;
    merge(
      basePath: string,
      adapterPath: string,
      outputPath: string,
    ): Promise<{ ok: boolean; error?: string }>;
    onEvent(cb: (msg: BridgeMsg) => void): () => void;
  };
  dialog: {
    openDir(): Promise<string | null>;
    openFile(
      filters?: { name: string; extensions: string[] }[],
    ): Promise<string | null>;
  };
  system: {
    diagnostics(): Promise<SystemDiagnostics>;
    gpuInfo(): Promise<GpuInfoResponse>;
  };
  onPaths(cb: (paths: AppPaths) => void): void;
}

interface SystemDiagnostics {
  ok: boolean;
  pythonConfigured: string;
  pythonResolved: string | null;
  pythonVersion: string | null;
  transformersVersion: string | null;
  stderr: string | null;
  raw: string | null;
}

interface GpuInfoResponse {
  ok: boolean;
  python: {
    torch_available?: boolean;
    cuda_available?: boolean;
    cuda_version?: string | null;
    device_count?: number;
    devices?: Array<{
      index: number;
      name: string;
      total_memory_gb: number;
      major: number;
      minor: number;
    }>;
    error?: string;
  } | null;
  nvidiaSmiAvailable: boolean;
  nvidia: Array<{
    name: string;
    driverVersion: string;
    memoryMb: number;
    temperatureC: number;
    utilizationPercent: number;
  }>;
  nvidiaError: string | null;
}

interface TrainConfig {
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

interface BridgeMsg {
  type: string;
  text?: string;
  step?: number;
  total?: number;
  message?: string;
  path?: string;
  code?: number;
  [key: string]: unknown;
}

interface HistoryEntry {
  id: string;
  type: "chat" | "image";
  timestamp: number;
  messages?: { role: string; content: string }[];
  prompt?: string;
  imagePath?: string;
  params?: Record<string, unknown>;
}

interface MediaFile {
  name: string;
  path: string;
  size: number;
  mtime: number;
}

interface ModelInfo {
  name: string;
  path: string;
  type: string;
  exists: boolean;
}

interface AppPaths {
  outputs: string;
  llmModel: string;
  imageModel: string;
  python: string;
}

interface Window {
  lavely: LavelyAPI;
}
