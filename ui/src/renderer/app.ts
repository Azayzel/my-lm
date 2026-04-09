// Lavely UI - Renderer process TypeScript
// Talks to main process through window.lavely (contextBridge)

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  llmReady: false,
  llmLoading: false,
  imageReady: false,
  imageLoading: false,
  generating: false,
  streaming: false,
  currentScreen: "chat",
  // conversation history for current session
  messages: [] as { role: string; content: string }[],
  paths: null as AppPaths | null,
};

// ─── DOM helpers ──────────────────────────────────────────────────────────────
function $<T extends HTMLElement>(
  sel: string,
  parent: ParentNode = document,
): T {
  return parent.querySelector<T>(sel)!;
}

function showToast(
  msg: string,
  kind: "success" | "error" | "info" = "info",
  ms = 3500,
) {
  const container = $("#toast-container");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), ms);
}

// ─── Navigation ───────────────────────────────────────────────────────────────
function activateScreen(name: string) {
  state.currentScreen = name;
  document
    .querySelectorAll<HTMLElement>(".screen")
    .forEach((s) => s.classList.remove("active"));
  document
    .querySelectorAll<HTMLElement>(".nav-btn")
    .forEach((b) => b.classList.remove("active"));
  const screen = document.getElementById(`screen-${name}`);
  if (screen) screen.classList.add("active");
  const navBtn = document.querySelector<HTMLElement>(
    `.nav-btn[data-screen="${name}"]`,
  );
  if (navBtn) navBtn.classList.add("active");

  if (name === "media") loadMediaGrid();
  if (name === "models") loadModelsList();
  if (name === "gpu") loadGpuInfo();
}

document
  .querySelectorAll<HTMLElement>(".nav-btn[data-screen]")
  .forEach((btn) => {
    btn.addEventListener("click", () => activateScreen(btn.dataset["screen"]!));
  });

// ─── Window controls ─────────────────────────────────────────────────────────
$("#btn-min").addEventListener("click", () => window.lavely.window.minimize());
$("#btn-max").addEventListener("click", () => window.lavely.window.maximize());
$("#btn-close").addEventListener("click", () => window.lavely.window.close());

// ─── App paths ────────────────────────────────────────────────────────────────
window.lavely.onPaths((paths) => {
  state.paths = paths;
  // Show which Python is being used in the sidebar tooltip
  const statusEl = document.querySelector(".model-status") as HTMLElement;
  if (statusEl) {
    const isVenv =
      paths.python.includes(".venv") || paths.python.includes("venv");
    const label = isVenv ? "✓ .venv" : "⚠ system python";
    const tip = document.createElement("div");
    tip.className = "ms-label";
    tip.style.marginTop = "6px";
    tip.style.fontSize = "10px";
    tip.style.color = isVenv ? "var(--success)" : "var(--warn)";
    tip.title = paths.python;
    tip.textContent = `Python: ${label}`;
    statusEl.appendChild(tip);
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// CHAT SCREEN
// ─────────────────────────────────────────────────────────────────────────────

const messagesEl = $("#messages");
const chatInput = $<HTMLTextAreaElement>("#chat-input");
const chatSendBtn = $<HTMLButtonElement>("#chat-send-btn");
const llmLoadBtn = $("#llm-load-btn");
const llmUnloadBtn = $("#llm-unload-btn");
const llmDot = $("#llm-dot");

// Range sliders
function bindRange(inputId: string, valId: string, decimals = 0) {
  const input = $<HTMLInputElement>(`#${inputId}`);
  const val = $(`#${valId}`);
  val.textContent = parseFloat(input.value).toFixed(decimals);
  input.addEventListener("input", () => {
    val.textContent = parseFloat(input.value).toFixed(decimals);
  });
}
bindRange("max-tokens", "max-tokens-val", 0);
bindRange("temperature", "temperature-val", 2);
bindRange("top-p", "top-p-val", 2);

function updateLLMUI() {
  llmDot.className =
    "dot" + (state.llmLoading ? " loading" : state.llmReady ? " on" : "");
  chatSendBtn.disabled = !state.llmReady || state.streaming;
  llmLoadBtn.textContent = state.llmLoading
    ? "Loading…"
    : state.llmReady
      ? "Reload"
      : "Load Model";
  (llmLoadBtn as HTMLElement).style.opacity = state.llmLoading ? "0.5" : "1";
  llmUnloadBtn.style.display = state.llmReady ? "" : "none";
}

updateLLMUI();

// Subscribe to LLM events
let currentAssistantBubble: HTMLElement | null = null;
let currentAssistantText = "";

window.lavely.llm.onEvent((msg: BridgeMsg) => {
  if (msg.type === "status") {
    showToast(msg["message"] as string, "info");
  } else if (msg.type === "ready") {
    state.llmReady = true;
    state.llmLoading = false;
    updateLLMUI();
    showToast("LLM model ready ✓", "success");
  } else if (msg.type === "token") {
    if (!currentAssistantBubble) {
      currentAssistantText = "";
      const msgEl = appendMessage("assistant", "");
      currentAssistantBubble = msgEl.querySelector(".bubble")!;
      currentAssistantBubble.classList.add("typing-cursor");
    }
    currentAssistantText += msg.text ?? "";
    currentAssistantBubble.textContent = currentAssistantText;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else if (msg.type === "done") {
    if (currentAssistantBubble) {
      currentAssistantBubble.classList.remove("typing-cursor");
      // Save to conversation state
      state.messages.push({ role: "assistant", content: currentAssistantText });
      currentAssistantBubble = null;
    }
    state.streaming = false;
    updateLLMUI();
  } else if (msg.type === "error") {
    state.streaming = false;
    state.llmLoading = false;
    updateLLMUI();
    showToast(`LLM error: ${msg["message"]}`, "error", 6000);
  } else if (msg.type === "exit") {
    state.llmReady = false;
    state.llmLoading = false;
    state.streaming = false;
    updateLLMUI();
  }
});

function appendMessage(role: "user" | "assistant", text: string): HTMLElement {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `
    <div class="avatar">${role === "user" ? "👤" : "✦"}</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function escapeHtml(s: string) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

llmLoadBtn.addEventListener("click", async () => {
  if (state.llmLoading) return;
  state.llmLoading = true;
  updateLLMUI();
  const res = await window.lavely.llm.start();
  if (!res.ok) {
    state.llmLoading = false;
    updateLLMUI();
    showToast(`Failed: ${res.error}`, "error", 6000);
  }
});

llmUnloadBtn.addEventListener("click", async () => {
  await window.lavely.llm.stop();
  state.llmReady = false;
  updateLLMUI();
  showToast("LLM unloaded", "info");
});

async function sendChat() {
  const text = chatInput.value.trim();
  if (!text || !state.llmReady || state.streaming) return;

  chatInput.value = "";
  chatInput.style.height = "";

  appendMessage("user", text);
  state.messages.push({ role: "user", content: text });
  state.streaming = true;
  currentAssistantBubble = null;
  updateLLMUI();

  const request = {
    messages: state.messages,
    system: ($("#system-prompt") as HTMLTextAreaElement).value,
    max_tokens: parseInt(($("#max-tokens") as HTMLInputElement).value),
    temperature: parseFloat(($("#temperature") as HTMLInputElement).value),
    top_p: parseFloat(($("#top-p") as HTMLInputElement).value),
  };

  const res = await window.lavely.llm.chat(request);
  if (!res.ok) {
    state.streaming = false;
    updateLLMUI();
    showToast(`Chat error: ${res.error}`, "error");
  }
}

chatSendBtn.addEventListener("click", sendChat);

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});

// Auto-resize textarea
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px";
});

$("#clear-chat-btn").addEventListener("click", () => {
  state.messages = [];
  messagesEl.innerHTML = `
    <div class="msg assistant">
      <div class="avatar">✦</div>
      <div class="bubble">Conversation cleared. How can I help?</div>
    </div>`;
});

// ─────────────────────────────────────────────────────────────────────────────
// IMAGE SCREEN
// ─────────────────────────────────────────────────────────────────────────────

const imgLoadBtn = $("#img-load-btn");
const imgUnloadBtn = $("#img-unload-btn");
const imgDot = $("#img-dot");
const generateBtn = $<HTMLButtonElement>("#generate-btn");
const genImage = $<HTMLImageElement>("#gen-image");
const imagePlaceholder = $("#image-placeholder");
const progressWrap = $("#progress-bar-wrap");
const progressFill = $("#progress-fill") as HTMLElement;
const progressSteps = $("#progress-steps");
const progressLabelText = $("#progress-label-text");

// Prompt builder live preview
function buildPrompt(): string {
  const head = ($("#pb-head") as HTMLInputElement).value.trim();
  const name = ($("#pb-name") as HTMLInputElement).value.trim();
  const position = ($("#pb-position") as HTMLInputElement).value.trim();
  const weights = ($("#pb-weights") as HTMLTextAreaElement).value.trim();
  return [head, name, position, weights].filter(Boolean).join(". ");
}

function updatePromptPreview() {
  $("#full-prompt-preview").textContent = buildPrompt();
}

["#pb-head", "#pb-name", "#pb-position", "#pb-weights"].forEach((id) => {
  $(id).addEventListener("input", updatePromptPreview);
});
updatePromptPreview();

// Range sliders for image
bindRange("img-steps", "img-steps-val", 0);
bindRange("img-cfg", "img-cfg-val", 1);

function updateImageUI() {
  imgDot.className =
    "dot" + (state.imageLoading ? " loading" : state.imageReady ? " on" : "");
  generateBtn.disabled = !state.imageReady || state.generating;
  imgLoadBtn.textContent = state.imageLoading
    ? "Loading…"
    : state.imageReady
      ? "Reload"
      : "Load Model";
  (imgLoadBtn as HTMLElement).style.opacity = state.imageLoading ? "0.5" : "1";
  imgUnloadBtn.style.display = state.imageReady ? "" : "none";
}

updateImageUI();

// Subscribe to image events
window.lavely.image.onEvent((msg: BridgeMsg) => {
  if (msg.type === "status") {
    showToast(msg["message"] as string, "info");
  } else if (msg.type === "ready") {
    state.imageReady = true;
    state.imageLoading = false;
    updateImageUI();
    showToast("Image model ready ✓", "success");
  } else if (msg.type === "progress") {
    const step = msg.step ?? 0;
    const total = msg.total ?? 1;
    const pct = Math.round((step / total) * 100);
    progressFill.style.width = `${pct}%`;
    progressSteps.textContent = `${step} / ${total}`;
    const label = (msg["message"] as string | undefined) ?? "Generating…";
    progressLabelText.textContent = label;
    progressWrap.style.display = "flex";
  } else if (msg.type === "done") {
    state.generating = false;
    updateImageUI();
    progressWrap.style.display = "none";
    progressFill.style.width = "0%";
    if (msg.path) {
      showImage(msg.path as string);
      // Save to history
      window.lavely.history.save({
        type: "image",
        prompt: buildPrompt(),
        imagePath: msg.path,
        params: {
          steps: parseInt(($("#img-steps") as HTMLInputElement).value),
          cfg: parseFloat(($("#img-cfg") as HTMLInputElement).value),
        },
      });
      showToast("Image generated ✓", "success");
    }
  } else if (msg.type === "error") {
    state.generating = false;
    state.imageLoading = false;
    updateImageUI();
    progressWrap.style.display = "none";
    showToast(`Image error: ${msg["message"]}`, "error", 6000);
  } else if (msg.type === "exit") {
    state.imageReady = false;
    state.imageLoading = false;
    state.generating = false;
    updateImageUI();
  }
});

function showImage(filePath: string) {
  genImage.src = `file://${filePath}?t=${Date.now()}`;
  genImage.style.display = "block";
  imagePlaceholder.style.display = "none";
}

imgLoadBtn.addEventListener("click", async () => {
  if (state.imageLoading) return;
  state.imageLoading = true;
  updateImageUI();
  const res = await window.lavely.image.start();
  if (!res.ok) {
    state.imageLoading = false;
    updateImageUI();
    showToast(`Failed: ${res.error}`, "error", 6000);
  }
});

imgUnloadBtn.addEventListener("click", async () => {
  await window.lavely.image.stop();
  state.imageReady = false;
  updateImageUI();
  showToast("Image model unloaded", "info");
});

generateBtn.addEventListener("click", async () => {
  if (!state.imageReady || state.generating) return;
  state.generating = true;
  updateImageUI();

  progressFill.style.width = "0%";
  progressSteps.textContent = "0 / 0";
  progressLabelText.textContent = "Starting…";
  progressWrap.style.display = "flex";

  const request = {
    prompt: buildPrompt(),
    negative_prompt: ($("#neg-prompt") as HTMLTextAreaElement).value,
    steps: parseInt(($("#img-steps") as HTMLInputElement).value),
    guidance: parseFloat(($("#img-cfg") as HTMLInputElement).value),
    width: parseInt(($("#img-width") as HTMLSelectElement).value),
    height: parseInt(($("#img-height") as HTMLSelectElement).value),
    upscale: ($("#img-upscale") as HTMLInputElement).checked,
  };

  const res = await window.lavely.image.generate(request);
  if (!res.ok) {
    state.generating = false;
    updateImageUI();
    progressWrap.style.display = "none";
    showToast(`Generate error: ${res.error}`, "error");
  }
});

// Click on generated image → open lightbox
genImage.addEventListener("click", () => {
  if (genImage.src)
    openLightbox(genImage.src, genImage.src.replace("file://", ""));
});

// ─────────────────────────────────────────────────────────────────────────────
// MEDIA SCREEN
// ─────────────────────────────────────────────────────────────────────────────

async function loadMediaGrid() {
  const grid = $("#media-grid");
  grid.innerHTML =
    '<div class="empty-state"><div class="es-icon">⏳</div><p>Loading…</p></div>';
  const files = await window.lavely.media.list();
  if (!files.length) {
    grid.innerHTML =
      '<div class="empty-state"><div class="es-icon">🖼️</div><p>No images yet. Generate something!</p></div>';
    return;
  }
  grid.innerHTML = "";
  files.forEach((f) => {
    const card = document.createElement("div");
    card.className = "media-card";
    card.innerHTML = `
      <img src="file://${f.path}" loading="lazy" alt="${f.name}" />
      <div class="media-card-info">
        <div class="media-card-name" title="${f.name}">${f.name}</div>
      </div>
      <div class="media-card-actions">
        <button title="Open" class="mc-open">↗</button>
        <button title="Show in folder" class="mc-show">📂</button>
        <button title="Delete" class="mc-del">🗑</button>
      </div>
    `;
    card
      .querySelector("img")!
      .addEventListener("click", () =>
        openLightbox(`file://${f.path}`, f.path),
      );
    card.querySelector(".mc-open")!.addEventListener("click", (e) => {
      e.stopPropagation();
      window.lavely.media.open(f.path);
    });
    card.querySelector(".mc-show")!.addEventListener("click", (e) => {
      e.stopPropagation();
      window.lavely.media.show(f.path);
    });
    card.querySelector(".mc-del")!.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete ${f.name}?`)) return;
      const res = await window.lavely.media.delete(f.path);
      if (res.ok) {
        card.remove();
        showToast("Deleted", "success");
      } else showToast(`Error: ${res.error}`, "error");
    });
    grid.appendChild(card);
  });
}

$("#media-refresh-btn").addEventListener("click", loadMediaGrid);

// ─────────────────────────────────────────────────────────────────────────────
// LIGHTBOX
// ─────────────────────────────────────────────────────────────────────────────

function openLightbox(src: string, filePath: string) {
  const lb = $("#lightbox");
  const img = $<HTMLImageElement>("#lightbox-img");
  const actions = $("#lightbox-actions");
  img.src = src;
  actions.innerHTML = `
    <button class="btn btn-ghost btn-sm" id="lb-open">Open in viewer</button>
    <button class="btn btn-ghost btn-sm" id="lb-folder">Show in folder</button>
  `;
  $("#lb-open").addEventListener("click", () =>
    window.lavely.media.open(filePath),
  );
  $("#lb-folder").addEventListener("click", () =>
    window.lavely.media.show(filePath),
  );
  lb.classList.add("open");
}

$("#lightbox-close").addEventListener("click", () =>
  $("#lightbox").classList.remove("open"),
);
$("#lightbox").addEventListener("click", (e) => {
  if (e.target === document.getElementById("lightbox"))
    $("#lightbox").classList.remove("open");
});

// ─────────────────────────────────────────────────────────────────────────────
// MODELS SCREEN
// ─────────────────────────────────────────────────────────────────────────────

const TYPE_ICONS: Record<string, string> = {
  llm: "🧠",
  image: "🎨",
  upscaler: "🔬",
  unknown: "📦",
};

async function loadModelsList() {
  const list = $("#models-list");
  list.innerHTML = "";
  const models = await window.lavely.models.list();
  if (!models.length) {
    list.innerHTML =
      '<div style="color:var(--text3);font-size:13px;">No models found in models/ directory.</div>';
    return;
  }
  models.forEach((m) => {
    const card = document.createElement("div");
    card.className = "model-card";
    card.innerHTML = `
      <div class="model-card-icon">${TYPE_ICONS[m.type] ?? "📦"}</div>
      <div class="model-card-info">
        <div class="mc-name">${m.name}</div>
        <div class="mc-path">${m.path}</div>
        <span class="mc-type">${m.type}</span>
      </div>
    `;
    list.appendChild(card);
  });
}

// Download
const dlLog = $<HTMLDivElement>("#download-log");
const diagnosticsLog = $<HTMLDivElement>("#diagnostics-log");

function logDownload(text: string) {
  dlLog.style.display = "block";
  dlLog.textContent += text + "\n";
  dlLog.scrollTop = dlLog.scrollHeight;
}

function logDiagnostics(text: string) {
  diagnosticsLog.style.display = "block";
  diagnosticsLog.textContent += text + "\n";
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}

window.lavely.models.onDownloadEvent((msg: BridgeMsg) => {
  if (msg.type === "status") logDownload(`ℹ ${msg["message"]}`);
  else if (msg.type === "done") {
    logDownload(`✓ Done! Saved to ${msg.path}`);
    showToast("Download complete ✓", "success");
    loadModelsList();
  } else if (msg.type === "error") {
    logDownload(`✗ Error: ${msg["message"]}`);
    showToast(`Download failed: ${msg["message"]}`, "error", 6000);
  } else if (msg.type === "stderr") {
    // Show HF progress lines
    logDownload(String(msg["text"]).trim());
  }
});

$("#dl-browse-btn").addEventListener("click", async () => {
  const dir = await window.lavely.dialog.openDir();
  if (dir) ($("#dl-target-dir") as HTMLInputElement).value = dir;
});

$("#dl-start-btn").addEventListener("click", async () => {
  const repoId = ($("#dl-repo-id") as HTMLInputElement).value.trim();
  const targetDir = ($("#dl-target-dir") as HTMLInputElement).value.trim();
  const type = ($("#dl-type") as HTMLSelectElement).value;
  if (!repoId || !targetDir) {
    showToast("Please fill in repo ID and target folder", "error");
    return;
  }
  dlLog.textContent = "";
  logDownload(`Starting download: ${repoId} → ${targetDir}`);
  await window.lavely.models.download(repoId, targetDir, type);
});

$("#run-diagnostics-btn").addEventListener("click", async () => {
  diagnosticsLog.textContent = "";
  logDiagnostics("Running environment diagnostics...");
  const info = await window.lavely.system.diagnostics();
  if (!info.ok) {
    logDiagnostics("✗ Diagnostics failed");
    if (info.stderr) logDiagnostics(`stderr: ${info.stderr}`);
    if (info.raw) logDiagnostics(`raw: ${info.raw}`);
    showToast("Diagnostics failed", "error", 5000);
    return;
  }
  logDiagnostics(`✓ Python configured: ${info.pythonConfigured}`);
  logDiagnostics(`✓ Python resolved: ${info.pythonResolved ?? "n/a"}`);
  logDiagnostics(`✓ Python version: ${info.pythonVersion ?? "n/a"}`);
  logDiagnostics(
    `✓ Transformers version: ${info.transformersVersion ?? "not installed"}`,
  );
  if (state.paths?.python && info.pythonResolved) {
    const same =
      state.paths.python.toLowerCase() === info.pythonResolved.toLowerCase();
    logDiagnostics(
      `✓ Interpreter match: ${same ? "yes" : "no (check env selection)"}`,
    );
  }
  showToast("Diagnostics complete ✓", "success");
});

// ─────────────────────────────────────────────────────────────────────────────
// GPU SCREEN
// ─────────────────────────────────────────────────────────────────────────────

const gpuContent = $<HTMLDivElement>("#gpu-content");
const gpuRaw = $<HTMLDivElement>("#gpu-raw");

async function loadGpuInfo() {
  gpuContent.innerHTML =
    '<div class="empty-state"><div class="es-icon">⏳</div><p>Loading GPU information…</p></div>';
  gpuRaw.textContent = "";

  const data = await window.lavely.system.gpuInfo();

  const cards: string[] = [];
  const py = data.python;
  cards.push(`
    <div class="gpu-card">
      <h3>PyTorch / CUDA</h3>
      <div class="kv">
        <div class="k">Torch</div><div class="v">${py?.torch_available ? "Available" : "Not available"}</div>
        <div class="k">CUDA</div><div class="v">${py?.cuda_available ? "Available" : "Not available"}</div>
        <div class="k">CUDA Version</div><div class="v">${py?.cuda_version ?? "n/a"}</div>
        <div class="k">Device Count</div><div class="v">${py?.device_count ?? 0}</div>
      </div>
    </div>
  `);

  if (py?.devices?.length) {
    py.devices.forEach((dev) => {
      cards.push(`
        <div class="gpu-card">
          <h3>GPU ${dev.index}: ${dev.name}</h3>
          <div class="kv">
            <div class="k">VRAM</div><div class="v">${dev.total_memory_gb} GB</div>
            <div class="k">Compute</div><div class="v">${dev.major}.${dev.minor}</div>
          </div>
        </div>
      `);
    });
  }

  if (data.nvidia?.length) {
    data.nvidia.forEach((gpu, index) => {
      cards.push(`
        <div class="gpu-card">
          <h3>NVIDIA-SMI ${index}</h3>
          <div class="kv">
            <div class="k">Name</div><div class="v">${gpu.name}</div>
            <div class="k">Driver</div><div class="v">${gpu.driverVersion}</div>
            <div class="k">Memory</div><div class="v">${gpu.memoryMb} MB</div>
            <div class="k">Temp</div><div class="v">${gpu.temperatureC} °C</div>
            <div class="k">Utilization</div><div class="v">${gpu.utilizationPercent}%</div>
          </div>
        </div>
      `);
    });
  }

  gpuContent.innerHTML = cards.length
    ? cards.join("\n")
    : '<div class="empty-state"><div class="es-icon">⚠</div><p>No GPU details found.</p></div>';

  gpuRaw.textContent = JSON.stringify(data, null, 2);
}

$("#gpu-refresh-btn").addEventListener("click", loadGpuInfo);

// ─────────────────────────────────────────────────────────────────────────────
// TRAIN SCREEN
// ─────────────────────────────────────────────────────────────────────────────

const trainState = {
  running: false,
  totalSteps: 0,
};

const trainBar = $<HTMLElement>("#train-bar");
const trainLog = $<HTMLElement>("#train-log");
const trainStartBtn = $<HTMLButtonElement>("#train-start-btn");
const trainStopBtn = $<HTMLButtonElement>("#train-stop-btn");
const trainMergeBtn = $<HTMLButtonElement>("#train-merge-btn");
const trainMStep = $("#train-m-step");
const trainMEpoch = $("#train-m-epoch");
const trainMLoss = $("#train-m-loss");
const trainMLr = $("#train-m-lr");

function trainLogLine(text: string, cls: "" | "ok" | "err" = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  trainLog.appendChild(line);
  trainLog.scrollTop = trainLog.scrollHeight;
}

function setTrainRunning(running: boolean) {
  trainState.running = running;
  trainStartBtn.disabled = running;
  trainStopBtn.disabled = !running;
  trainStartBtn.textContent = running ? "Training…" : "▶ Start Training";
}

// Browse handlers — directory for model/output, file for dataset
$("#train-model-browse").addEventListener("click", async () => {
  const dir = await window.lavely.dialog.openDir();
  if (dir) ($("#train-model-path") as HTMLInputElement).value = dir;
});
$("#train-output-browse").addEventListener("click", async () => {
  const dir = await window.lavely.dialog.openDir();
  if (dir) ($("#train-output-path") as HTMLInputElement).value = dir;
});
$("#train-dataset-browse").addEventListener("click", async () => {
  const file = await window.lavely.dialog.openFile([
    { name: "Dataset", extensions: ["jsonl", "json"] },
    { name: "All Files", extensions: ["*"] },
  ]);
  if (file) ($("#train-dataset-path") as HTMLInputElement).value = file;
});

// Prefill model path from app paths
window.lavely.onPaths((paths) => {
  const modelInput = $<HTMLInputElement>("#train-model-path");
  if (!modelInput.value && paths.llmModel) modelInput.value = paths.llmModel;
});

function readTrainConfig(): TrainConfig {
  const v = (id: string) => ($(`#${id}`) as HTMLInputElement).value;
  const n = (id: string) => parseFloat(v(id));
  const i = (id: string) => parseInt(v(id), 10);
  return {
    model_path: v("train-model-path").trim(),
    dataset_path: v("train-dataset-path").trim(),
    output_dir: v("train-output-path").trim() || "models/lavely-lm-lora",
    epochs: i("train-epochs"),
    batch_size: i("train-batch"),
    grad_accum: i("train-grad-accum"),
    lr: n("train-lr"),
    lora_r: i("train-lora-r"),
    lora_alpha: i("train-lora-alpha"),
    lora_dropout: n("train-lora-dropout"),
    max_seq_len: i("train-max-seq"),
    logging_steps: i("train-log-steps"),
    use_4bit: ($("#train-use-4bit") as HTMLInputElement).checked,
  };
}

trainStartBtn.addEventListener("click", async () => {
  const cfg = readTrainConfig();
  if (!cfg.model_path || !cfg.dataset_path) {
    showToast("Please set a base model and dataset path", "error");
    return;
  }
  trainLog.innerHTML = "";
  trainBar.style.width = "0%";
  trainMStep.textContent = "—";
  trainMEpoch.textContent = "—";
  trainMLoss.textContent = "—";
  trainMLr.textContent = "—";
  trainLogLine(`Starting training with config:`);
  trainLogLine(JSON.stringify(cfg, null, 2));

  setTrainRunning(true);
  const res = await window.lavely.train.start(cfg);
  if (!res.ok) {
    trainLogLine(`✗ Failed to start: ${res.error}`, "err");
    showToast(`Training failed to start: ${res.error}`, "error", 6000);
    setTrainRunning(false);
  }
});

trainStopBtn.addEventListener("click", async () => {
  await window.lavely.train.stop();
  trainLogLine("■ Training stopped by user", "err");
  setTrainRunning(false);
});

trainMergeBtn.addEventListener("click", async () => {
  const cfg = readTrainConfig();
  if (!cfg.model_path || !cfg.output_dir) {
    showToast("Set base model + adapter output directory first", "error");
    return;
  }
  const mergedPath = cfg.output_dir + "-merged";
  trainLogLine(`Merging adapter from ${cfg.output_dir} into ${mergedPath}...`);
  const res = await window.lavely.train.merge(
    cfg.model_path,
    cfg.output_dir,
    mergedPath,
  );
  if (res.ok) {
    trainLogLine(`✓ Merged model saved to ${mergedPath}`, "ok");
    showToast("Merge complete ✓", "success");
  } else {
    trainLogLine(`✗ Merge failed: ${res.error}`, "err");
    showToast(`Merge failed: ${res.error}`, "error", 6000);
  }
});

window.lavely.train.onEvent((msg: BridgeMsg) => {
  switch (msg.type) {
    case "status":
      trainLogLine(`ℹ ${msg["message"]}`);
      break;
    case "config":
      trainState.totalSteps = Number(msg["total_steps"]) || 0;
      trainLogLine(
        `Config — model: ${msg["model"]}, dataset: ${msg["dataset"]}, samples: ${msg["samples"]}, total steps: ${msg["total_steps"]}`,
      );
      break;
    case "step": {
      const step = Number(msg.step) || 0;
      const total = Number(msg.total) || trainState.totalSteps || 1;
      const pct = Math.min(100, Math.max(0, (step / total) * 100));
      trainBar.style.width = `${pct.toFixed(1)}%`;
      trainMStep.textContent = `${step} / ${total}`;
      if (msg["epoch"] !== undefined)
        trainMEpoch.textContent = Number(msg["epoch"]).toFixed(2);
      if (msg["loss"] !== undefined)
        trainMLoss.textContent = Number(msg["loss"]).toFixed(4);
      if (msg["lr"] !== undefined)
        trainMLr.textContent = Number(msg["lr"]).toExponential(2);
      break;
    }
    case "epoch":
      trainLogLine(`▶ Epoch ${msg["epoch"]} / ${msg["total"]}`);
      break;
    case "log":
      trainLogLine(String(msg["text"] ?? ""));
      break;
    case "stderr":
      // quiet filter — only show interesting stderr lines
      {
        const t = String(msg["text"] ?? "");
        if (
          t &&
          !t.includes("FutureWarning") &&
          !t.includes("UserWarning") &&
          !t.includes("warnings.warn")
        ) {
          trainLogLine(t);
        }
      }
      break;
    case "done":
      trainLogLine(
        `✓ Training complete. Adapter saved to ${msg["output"]}`,
        "ok",
      );
      showToast("Training complete ✓", "success");
      trainBar.style.width = "100%";
      setTrainRunning(false);
      break;
    case "error":
      trainLogLine(`✗ ${msg["message"]}`, "err");
      showToast(`Training error: ${msg["message"]}`, "error", 6000);
      setTrainRunning(false);
      break;
    case "exit":
      if (trainState.running) {
        trainLogLine(`Process exited with code ${msg["code"]}`);
        setTrainRunning(false);
      }
      break;
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
activateScreen("chat");
