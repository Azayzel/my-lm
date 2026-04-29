import { contextBridge, ipcRenderer } from "electron";

// Expose a safe, typed API to the renderer process
contextBridge.exposeInMainWorld("My", {
  // Window controls
  window: {
    minimize: () => ipcRenderer.send("window:minimize"),
    maximize: () => ipcRenderer.send("window:maximize"),
    close: () => ipcRenderer.send("window:close"),
  },

  // LLM
  llm: {
    start: (modelPath?: string) => ipcRenderer.invoke("llm:start", modelPath),
    chat: (request: object) => ipcRenderer.invoke("llm:chat", request),
    stop: () => ipcRenderer.invoke("llm:stop"),
    status: () => ipcRenderer.invoke("llm:status"),
    onEvent: (cb: (msg: object) => void) => {
      const handler = (_: unknown, msg: object) => cb(msg);
      ipcRenderer.on("llm:event", handler);
      return () => ipcRenderer.removeListener("llm:event", handler);
    },
  },

  // Image generation
  image: {
    start: (modelPath?: string) => ipcRenderer.invoke("image:start", modelPath),
    generate: (request: object) =>
      ipcRenderer.invoke("image:generate", request),
    stop: () => ipcRenderer.invoke("image:stop"),
    status: () => ipcRenderer.invoke("image:status"),
    onEvent: (cb: (msg: object) => void) => {
      const handler = (_: unknown, msg: object) => cb(msg);
      ipcRenderer.on("image:event", handler);
      return () => ipcRenderer.removeListener("image:event", handler);
    },
  },

  // History
  history: {
    get: () => ipcRenderer.invoke("history:get"),
    save: (entry: object) => ipcRenderer.invoke("history:save", entry),
    clear: () => ipcRenderer.invoke("history:clear"),
    delete: (id: string) => ipcRenderer.invoke("history:delete", id),
  },

  // BookMind (RAG)
  books: {
    start: () => ipcRenderer.invoke("books:start"),
    query: (request: object) => ipcRenderer.invoke("books:query", request),
    stop: () => ipcRenderer.invoke("books:stop"),
    status: () => ipcRenderer.invoke("books:status"),
    onEvent: (cb: (msg: object) => void) => {
      const handler = (_: unknown, msg: object) => cb(msg);
      ipcRenderer.on("books:event", handler);
      return () => ipcRenderer.removeListener("books:event", handler);
    },
  },

  // Curated model catalog
  catalog: {
    list: (vramGb?: number) => ipcRenderer.invoke("catalog:list", vramGb),
  },

  // Saved prompts
  prompts: {
    list: () => ipcRenderer.invoke("prompts:list"),
    save: (entry: object) => ipcRenderer.invoke("prompts:save", entry),
    update: (id: string, patch: object) =>
      ipcRenderer.invoke("prompts:update", id, patch),
    delete: (id: string) => ipcRenderer.invoke("prompts:delete", id),
  },

  // Media
  media: {
    list: (subdir?: string) => ipcRenderer.invoke("media:list", subdir),
    createFolder: (name: string) =>
      ipcRenderer.invoke("media:createFolder", name),
    deleteFolder: (relPath: string) =>
      ipcRenderer.invoke("media:deleteFolder", relPath),
    move: (filePath: string, destFolder: string) =>
      ipcRenderer.invoke("media:move", filePath, destFolder),
    delete: (filePath: string) => ipcRenderer.invoke("media:delete", filePath),
    open: (filePath: string) => ipcRenderer.invoke("media:open", filePath),
    show: (filePath: string) => ipcRenderer.invoke("media:show", filePath),
  },

  // Models
  models: {
    list: () => ipcRenderer.invoke("models:list"),
    download: (repoId: string, targetDir: string, type: string) =>
      ipcRenderer.invoke("models:download", repoId, targetDir, type),
    exists: (modelPath: string) =>
      ipcRenderer.invoke("models:exists", modelPath),
    onDownloadEvent: (cb: (msg: object) => void) => {
      const handler = (_: unknown, msg: object) => cb(msg);
      ipcRenderer.on("models:download:event", handler);
      return () => ipcRenderer.removeListener("models:download:event", handler);
    },
  },

  // Training
  train: {
    start: (config: object) => ipcRenderer.invoke("train:start", config),
    stop: () => ipcRenderer.invoke("train:stop"),
    status: () => ipcRenderer.invoke("train:status"),
    merge: (basePath: string, adapterPath: string, outputPath: string) =>
      ipcRenderer.invoke("train:merge", basePath, adapterPath, outputPath),
    onEvent: (cb: (msg: object) => void) => {
      const handler = (_: unknown, msg: object) => cb(msg);
      ipcRenderer.on("train:event", handler);
      return () => ipcRenderer.removeListener("train:event", handler);
    },
  },

  // Benchmarking (agent_bench.py)
  bench: {
    start: (config: object) => ipcRenderer.invoke("bench:start", config),
    stop: () => ipcRenderer.invoke("bench:stop"),
    status: () => ipcRenderer.invoke("bench:status"),
    listResults: () => ipcRenderer.invoke("bench:listResults"),
    getResult: (filePath: string) =>
      ipcRenderer.invoke("bench:getResult", filePath),
    openResultsDir: () => ipcRenderer.invoke("bench:openResultsDir"),
    onEvent: (cb: (msg: object) => void) => {
      const handler = (_: unknown, msg: object) => cb(msg);
      ipcRenderer.on("bench:event", handler);
      return () => ipcRenderer.removeListener("bench:event", handler);
    },
  },

  // Config / Settings
  config: {
    get: () => ipcRenderer.invoke("config:get"),
    set: (patch: object) => ipcRenderer.invoke("config:set", patch),
    testMongo: (uri: string, dbName: string) =>
      ipcRenderer.invoke("config:testMongo", uri, dbName),
  },

  // System
  dialog: {
    openDir: () => ipcRenderer.invoke("dialog:openDir"),
    openFile: (filters?: { name: string; extensions: string[] }[]) =>
      ipcRenderer.invoke("dialog:openFile", filters),
  },

  system: {
    diagnostics: () => ipcRenderer.invoke("system:diagnostics"),
    gpuInfo: () => ipcRenderer.invoke("system:gpuInfo"),
  },

  onPaths: (
    cb: (paths: {
      outputs: string;
      llmModel: string;
      imageModel: string;
      python: string;
    }) => void,
  ) => {
    ipcRenderer.on("app:paths", (_: unknown, paths) => cb(paths));
  },
});
