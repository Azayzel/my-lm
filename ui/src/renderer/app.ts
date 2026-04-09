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

// Electron disables window.prompt() in renderer — this is our replacement.
function askInput(
  title: string,
  body = "",
  placeholder = "",
  initial = "",
): Promise<string | null> {
  return new Promise((resolve) => {
    const modal = $<HTMLElement>("#input-modal");
    const titleEl = $<HTMLElement>("#input-modal-title");
    const bodyEl = $<HTMLElement>("#input-modal-body");
    const input = $<HTMLInputElement>("#input-modal-input");
    const okBtn = $<HTMLButtonElement>("#input-modal-ok");
    const cancelBtn = $<HTMLButtonElement>("#input-modal-cancel");

    titleEl.textContent = title;
    bodyEl.textContent = body;
    input.placeholder = placeholder;
    input.value = initial;
    modal.classList.add("open");
    setTimeout(() => input.focus(), 0);

    const cleanup = () => {
      modal.classList.remove("open");
      okBtn.removeEventListener("click", onOk);
      cancelBtn.removeEventListener("click", onCancel);
      input.removeEventListener("keydown", onKey);
    };
    const onOk = () => {
      const v = input.value.trim();
      cleanup();
      resolve(v || null);
    };
    const onCancel = () => {
      cleanup();
      resolve(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Enter") onOk();
      else if (e.key === "Escape") onCancel();
    };

    okBtn.addEventListener("click", onOk);
    cancelBtn.addEventListener("click", onCancel);
    input.addEventListener("keydown", onKey);
  });
}

function askConfirm(title: string, body = ""): Promise<boolean> {
  return new Promise((resolve) => {
    const modal = $<HTMLElement>("#input-modal");
    const titleEl = $<HTMLElement>("#input-modal-title");
    const bodyEl = $<HTMLElement>("#input-modal-body");
    const input = $<HTMLInputElement>("#input-modal-input");
    const okBtn = $<HTMLButtonElement>("#input-modal-ok");
    const cancelBtn = $<HTMLButtonElement>("#input-modal-cancel");

    titleEl.textContent = title;
    bodyEl.textContent = body;
    input.style.display = "none";
    okBtn.textContent = "Confirm";
    modal.classList.add("open");
    setTimeout(() => okBtn.focus(), 0);

    const cleanup = () => {
      modal.classList.remove("open");
      input.style.display = "";
      okBtn.textContent = "OK";
      okBtn.removeEventListener("click", onOk);
      cancelBtn.removeEventListener("click", onCancel);
      document.removeEventListener("keydown", onKey);
    };
    const onOk = () => {
      cleanup();
      resolve(true);
    };
    const onCancel = () => {
      cleanup();
      resolve(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (!modal.classList.contains("open")) return;
      if (e.key === "Enter") onOk();
      else if (e.key === "Escape") onCancel();
    };

    okBtn.addEventListener("click", onOk);
    cancelBtn.addEventListener("click", onCancel);
    document.addEventListener("keydown", onKey);
  });
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
  if (name === "models") {
    loadModelsList();
    loadCatalog();
  }
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
const llmStopBtn = $("#llm-stop-btn");
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
  const llmActive = state.llmLoading || state.llmReady || state.streaming;
  llmDot.className =
    "dot" + (state.llmLoading ? " loading" : state.llmReady ? " on" : "");
  chatSendBtn.disabled = !state.llmReady || state.streaming;
  llmLoadBtn.textContent = state.llmLoading
    ? "Loading…"
    : state.llmReady
      ? "Reload"
      : "Load Model";
  (llmLoadBtn as HTMLElement).style.opacity = state.llmLoading ? "0.5" : "1";
  llmStopBtn.style.display = llmActive ? "" : "none";
  llmUnloadBtn.style.display = state.llmReady && !state.streaming ? "" : "none";
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
    if (currentAssistantBubble) {
      currentAssistantBubble.classList.remove("typing-cursor");
      currentAssistantBubble = null;
    }
    updateLLMUI();
    showToast(`LLM error: ${msg["message"]}`, "error", 6000);
  } else if (msg.type === "exit") {
    state.llmReady = false;
    state.llmLoading = false;
    state.streaming = false;
    if (currentAssistantBubble) {
      currentAssistantBubble.classList.remove("typing-cursor");
      currentAssistantBubble = null;
    }
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

llmStopBtn.addEventListener("click", async () => {
  await window.lavely.llm.stop();
  state.llmReady = false;
  state.llmLoading = false;
  state.streaming = false;
  if (currentAssistantBubble) {
    currentAssistantBubble.classList.remove("typing-cursor");
    currentAssistantBubble = null;
  }
  updateLLMUI();
  showToast("LLM process stopped", "info");
});

llmUnloadBtn.addEventListener("click", async () => {
  await window.lavely.llm.stop();
  state.llmReady = false;
  state.llmLoading = false;
  state.streaming = false;
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
const imgStopBtn = $("#img-stop-btn");
const imgUnloadBtn = $("#img-unload-btn");
const imgDot = $("#img-dot");
const generateBtn = $<HTMLButtonElement>("#generate-btn");
const generateStopBtn = $("#generate-stop-btn");
const genImage = $<HTMLImageElement>("#gen-image");
const imagePlaceholder = $("#image-placeholder");
const progressWrap = $("#progress-bar-wrap");
const progressFill = $("#progress-fill") as HTMLElement;
const progressSteps = $("#progress-steps");
const progressLabelText = $("#progress-label-text");
const progressEta = $("#progress-eta");

// ETA tracking — stores (step, timestamp_ms) for the last few steps so we can
// compute a moving-average step duration.
const etaState = {
  startTs: 0,
  stepTimes: [] as { step: number; ts: number }[],
};

function resetEta() {
  etaState.startTs = Date.now();
  etaState.stepTimes = [];
  progressEta.textContent = "";
}

function formatDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return `${m}m ${r}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function updateEta(step: number, total: number) {
  const now = Date.now();
  etaState.stepTimes.push({ step, ts: now });
  // Keep the last 6 samples for a stable moving average
  if (etaState.stepTimes.length > 6) etaState.stepTimes.shift();

  if (etaState.stepTimes.length < 2 || step >= total) {
    progressEta.textContent = "";
    return;
  }

  const first = etaState.stepTimes[0];
  const last = etaState.stepTimes[etaState.stepTimes.length - 1];
  const stepsDone = last.step - first.step;
  const elapsed = (last.ts - first.ts) / 1000; // seconds
  if (stepsDone <= 0 || elapsed <= 0) return;

  const secPerStep = elapsed / stepsDone;
  const remaining = (total - step) * secPerStep;
  progressEta.textContent = `~${formatDuration(remaining)} left`;
}

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
bindRange("img-face-fix-strength", "img-face-fix-strength-val", 2);

function updateImageUI() {
  const imageActive =
    state.imageLoading || state.imageReady || state.generating;
  imgDot.className =
    "dot" + (state.imageLoading ? " loading" : state.imageReady ? " on" : "");
  generateBtn.disabled = !state.imageReady || state.generating;
  imgLoadBtn.textContent = state.imageLoading
    ? "Loading…"
    : state.imageReady
      ? "Reload"
      : "Load Model";
  (imgLoadBtn as HTMLElement).style.opacity = state.imageLoading ? "0.5" : "1";
  imgStopBtn.style.display = imageActive ? "" : "none";
  imgUnloadBtn.style.display =
    state.imageReady && !state.generating ? "" : "none";
  generateStopBtn.style.display = state.generating ? "" : "none";
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
    updateEta(step, total);
    // Streaming preview from TAESDXL
    const preview = msg["preview"];
    if (typeof preview === "string" && preview.length > 0) {
      genImage.src = `data:image/png;base64,${preview}`;
      genImage.style.display = "block";
      imagePlaceholder.style.display = "none";
    }
  } else if (msg.type === "done") {
    state.generating = false;
    updateImageUI();
    progressWrap.style.display = "none";
    progressFill.style.width = "0%";
    progressEta.textContent = "";
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
    progressFill.style.width = "0%";
    showToast(`Image error: ${msg["message"]}`, "error", 6000);
  } else if (msg.type === "exit") {
    state.imageReady = false;
    state.imageLoading = false;
    state.generating = false;
    progressWrap.style.display = "none";
    progressFill.style.width = "0%";
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

imgStopBtn.addEventListener("click", async () => {
  await window.lavely.image.stop();
  state.imageReady = false;
  state.imageLoading = false;
  state.generating = false;
  progressWrap.style.display = "none";
  progressFill.style.width = "0%";
  updateImageUI();
  showToast("Image process stopped", "info");
});

imgUnloadBtn.addEventListener("click", async () => {
  await window.lavely.image.stop();
  state.imageReady = false;
  state.imageLoading = false;
  state.generating = false;
  progressWrap.style.display = "none";
  progressFill.style.width = "0%";
  updateImageUI();
  showToast("Image model unloaded", "info");
});

generateStopBtn.addEventListener("click", async () => {
  await window.lavely.image.stop();
  state.imageReady = false;
  state.imageLoading = false;
  state.generating = false;
  progressWrap.style.display = "none";
  progressFill.style.width = "0%";
  updateImageUI();
  showToast("Generation stopped", "info");
});

generateBtn.addEventListener("click", async () => {
  if (!state.imageReady || state.generating) return;
  state.generating = true;
  updateImageUI();

  progressFill.style.width = "0%";
  progressSteps.textContent = "0 / 0";
  progressLabelText.textContent = "Starting…";
  progressWrap.style.display = "flex";
  resetEta();

  const request = {
    prompt: buildPrompt(),
    negative_prompt: ($("#neg-prompt") as HTMLTextAreaElement).value,
    steps: parseInt(($("#img-steps") as HTMLInputElement).value),
    guidance: parseFloat(($("#img-cfg") as HTMLInputElement).value),
    width: parseInt(($("#img-width") as HTMLSelectElement).value),
    height: parseInt(($("#img-height") as HTMLSelectElement).value),
    upscale: ($("#img-upscale") as HTMLInputElement).checked,
    face_fix: ($("#img-face-fix") as HTMLInputElement).checked,
    face_fix_strength: parseFloat(
      ($("#img-face-fix-strength") as HTMLInputElement).value,
    ),
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

// ─────────────────────────────────────────────────────────────────────────────
// MODEL CATALOG
// ─────────────────────────────────────────────────────────────────────────────

const catalogState = {
  vramGb: 0,
  entries: [] as CatalogEntry[],
};

const catalogGrid = $<HTMLDivElement>("#catalog-grid");
const catalogVramBadge = $<HTMLElement>("#catalog-vram-badge");
const catalogShowAll = $<HTMLInputElement>("#catalog-show-all");
const catalogCategoryFilter = $<HTMLSelectElement>("#catalog-category-filter");

async function detectGpuVram(): Promise<number> {
  try {
    const info = await window.lavely.system.gpuInfo();
    const dev = info.python?.devices?.[0];
    if (dev?.total_memory_gb) return dev.total_memory_gb;
    const sm = info.nvidia?.[0];
    if (sm?.memoryMb) return sm.memoryMb / 1024;
  } catch {
    /* ignore */
  }
  return 0;
}

const CAT_LABEL: Record<CatalogEntry["category"], string> = {
  llm: "LLM",
  image: "Image",
  upscaler: "Upscaler",
  face: "Face",
  vae: "VAE",
};

function renderCatalog() {
  catalogGrid.innerHTML = "";
  const showAll = catalogShowAll.checked;
  const categoryFilter = catalogCategoryFilter.value;
  const vram = catalogState.vramGb;

  const visible = catalogState.entries.filter((e) => {
    if (categoryFilter && e.category !== categoryFilter) return false;
    if (!showAll && vram > 0 && e.minVramGb > vram) return false;
    return true;
  });

  if (!visible.length) {
    catalogGrid.innerHTML =
      '<div style="color:var(--text3);font-size:12px;grid-column:1/-1;padding:16px;text-align:center;">No models match the current filter.</div>';
    return;
  }

  visible.forEach((e) => {
    const fits = vram === 0 || e.minVramGb <= vram;
    const card = document.createElement("div");
    card.className = "catalog-card";
    if (e.installed) card.classList.add("installed");
    if (!fits) card.classList.add("too-big");

    const tags = [`<span class="cc-tag cat-${e.category}">${CAT_LABEL[e.category]}</span>`];
    tags.push(`<span class="cc-tag vram">${e.minVramGb}GB+</span>`);
    if (e.tags.includes("recommended")) {
      tags.push('<span class="cc-tag recommended">Recommended</span>');
    }

    const btnLabel = e.installed ? "✓ Installed" : "⬇ Install";
    const btnDisabled = e.installed ? "disabled" : "";

    card.innerHTML = `
      <div class="cc-name">${e.name}</div>
      <div class="cc-repo">${e.repoId}</div>
      <div class="cc-desc">${e.description}</div>
      <div class="cc-meta">${tags.join("")}</div>
      <div class="cc-footer">
        <span class="cc-size">${e.sizeGb.toFixed(1)} GB</span>
        <button class="btn btn-primary cc-install-btn" data-id="${e.id}" ${btnDisabled}>
          ${btnLabel}
        </button>
      </div>
    `;

    const btn = card.querySelector<HTMLButtonElement>(".cc-install-btn")!;
    btn.addEventListener("click", async () => {
      if (e.installed) return;
      btn.disabled = true;
      btn.textContent = "Downloading…";
      const type =
        e.category === "llm" ? "llm" : e.category === "image" ? "image" : "other";
      logDownload(`▶ Installing ${e.name} (${e.repoId}) → models/${e.targetDir}`);
      await window.lavely.models.download(e.repoId, e.targetDir, type);
    });

    catalogGrid.appendChild(card);
  });
}

async function loadCatalog() {
  if (catalogState.vramGb === 0) {
    catalogState.vramGb = await detectGpuVram();
    if (catalogState.vramGb > 0) {
      catalogVramBadge.textContent = `Detected: ${catalogState.vramGb.toFixed(1)} GB VRAM`;
      catalogVramBadge.classList.add("ok");
    } else {
      catalogVramBadge.textContent = "GPU not detected — showing all";
      catalogShowAll.checked = true;
    }
  }
  catalogState.entries = await window.lavely.catalog.list(catalogState.vramGb || undefined);
  // Re-fetch with no filter if "show all" — the filtering is done client-side
  // so we can toggle instantly without a round-trip.
  if (catalogShowAll.checked) {
    catalogState.entries = await window.lavely.catalog.list();
  }
  renderCatalog();
}

catalogShowAll.addEventListener("change", async () => {
  // When toggling to "show all", fetch the full list
  catalogState.entries = await window.lavely.catalog.list(
    catalogShowAll.checked ? undefined : catalogState.vramGb || undefined,
  );
  renderCatalog();
});
catalogCategoryFilter.addEventListener("change", renderCatalog);

// Refresh catalog state whenever a download completes (installed flag flips)
window.lavely.models.onDownloadEvent((msg: BridgeMsg) => {
  if (msg.type === "done") {
    loadCatalog();
  }
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

// ─────────────────────────────────────────────────────────────────────────────
// SAVED PROMPTS
// ─────────────────────────────────────────────────────────────────────────────

const savedPromptsSelect = $<HTMLSelectElement>("#saved-prompts-select");
const savePromptBtn = $<HTMLButtonElement>("#save-prompt-btn");
const loadPromptBtn = $<HTMLButtonElement>("#load-prompt-btn");
const deletePromptBtn = $<HTMLButtonElement>("#delete-prompt-btn");

let savedPromptsCache: SavedPrompt[] = [];

async function refreshSavedPrompts() {
  savedPromptsCache = await window.lavely.prompts.list();
  const prev = savedPromptsSelect.value;
  savedPromptsSelect.innerHTML =
    '<option value="">— select a saved prompt —</option>';
  for (const p of savedPromptsCache) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.name;
    savedPromptsSelect.appendChild(opt);
  }
  // restore selection if still present
  if (savedPromptsCache.some((p) => p.id === prev)) {
    savedPromptsSelect.value = prev;
  }
}

savePromptBtn.addEventListener("click", async () => {
  const name = await askInput(
    "Save prompt",
    "Give this prompt a name so you can recall it later.",
    "e.g. Cinematic portrait",
  );
  if (!name) return;
  const entry = {
    name,
    prompt: buildPrompt(),
    negative_prompt: ($("#neg-prompt") as HTMLTextAreaElement).value,
    params: {
      head: ($("#pb-head") as HTMLInputElement).value,
      name: ($("#pb-name") as HTMLInputElement).value,
      position: ($("#pb-position") as HTMLInputElement).value,
      weights: ($("#pb-weights") as HTMLTextAreaElement).value,
      steps: parseInt(($("#img-steps") as HTMLInputElement).value),
      guidance: parseFloat(($("#img-cfg") as HTMLInputElement).value),
      width: parseInt(($("#img-width") as HTMLSelectElement).value),
      height: parseInt(($("#img-height") as HTMLSelectElement).value),
      upscale: ($("#img-upscale") as HTMLInputElement).checked,
      face_fix: ($("#img-face-fix") as HTMLInputElement).checked,
      face_fix_strength: parseFloat(
        ($("#img-face-fix-strength") as HTMLInputElement).value,
      ),
    },
  };
  await window.lavely.prompts.save(entry);
  await refreshSavedPrompts();
  showToast(`Saved prompt "${name}" ✓`, "success");
});

loadPromptBtn.addEventListener("click", () => {
  const id = savedPromptsSelect.value;
  if (!id) {
    showToast("Select a prompt first", "info");
    return;
  }
  const entry = savedPromptsCache.find((p) => p.id === id);
  if (!entry) return;

  const params = (entry.params ?? {}) as Record<string, unknown>;
  const setInput = (sel: string, value: unknown) => {
    const el = $<HTMLInputElement | HTMLTextAreaElement>(sel);
    if (value !== undefined && value !== null) el.value = String(value);
  };
  const setCheck = (sel: string, value: unknown) => {
    const el = $<HTMLInputElement>(sel);
    if (typeof value === "boolean") el.checked = value;
  };

  // Restore prompt builder fields if present, else dump the raw prompt into the head
  if (params["head"] !== undefined) {
    setInput("#pb-head", params["head"]);
    setInput("#pb-name", params["name"]);
    setInput("#pb-position", params["position"]);
    setInput("#pb-weights", params["weights"]);
  } else {
    setInput("#pb-head", entry.prompt);
    setInput("#pb-name", "");
    setInput("#pb-position", "");
    setInput("#pb-weights", "");
  }
  setInput("#neg-prompt", entry.negative_prompt);

  if (params["steps"] !== undefined) setInput("#img-steps", params["steps"]);
  if (params["guidance"] !== undefined)
    setInput("#img-cfg", params["guidance"]);
  if (params["width"] !== undefined) setInput("#img-width", params["width"]);
  if (params["height"] !== undefined) setInput("#img-height", params["height"]);
  setCheck("#img-upscale", params["upscale"]);
  setCheck("#img-face-fix", params["face_fix"]);
  if (params["face_fix_strength"] !== undefined) {
    setInput("#img-face-fix-strength", params["face_fix_strength"]);
  }

  // Trigger range displays + prompt preview
  ["#img-steps", "#img-cfg", "#img-face-fix-strength"].forEach((sel) =>
    $(sel).dispatchEvent(new Event("input")),
  );
  updatePromptPreview();
  showToast(`Loaded "${entry.name}"`, "success");
});

deletePromptBtn.addEventListener("click", async () => {
  const id = savedPromptsSelect.value;
  if (!id) {
    showToast("Select a prompt first", "info");
    return;
  }
  const entry = savedPromptsCache.find((p) => p.id === id);
  if (!entry) return;
  const ok = await askConfirm(
    "Delete saved prompt?",
    `"${entry.name}" will be permanently removed.`,
  );
  if (!ok) return;
  await window.lavely.prompts.delete(id);
  await refreshSavedPrompts();
  showToast("Prompt deleted", "info");
});

refreshSavedPrompts();

// ─────────────────────────────────────────────────────────────────────────────
// FIRST-RUN ESSENTIALS MODAL
// ─────────────────────────────────────────────────────────────────────────────

interface Essential {
  id: string;
  name: string;
  sub: string;
  icon: string;
  sizeGb: number;
  repoId: string;
  targetDir: string;
  type: string;
}

const ESSENTIALS: Essential[] = [
  {
    id: "qwen25-3b",
    name: "Qwen 2.5 3B Instruct",
    sub: "Default chat LLM",
    icon: "💬",
    sizeGb: 6.2,
    repoId: "Qwen/Qwen2.5-3B-Instruct",
    targetDir: "qwen2.5-3b",
    type: "llm",
  },
  {
    id: "realvis-v4",
    name: "RealVisXL V4.0",
    sub: "Photorealistic SDXL image model",
    icon: "🎨",
    sizeGb: 6.5,
    repoId: "SG161222/RealVisXL_V4.0",
    targetDir: "realvisxl-v4",
    type: "image",
  },
  {
    id: "4x-ultrasharp",
    name: "4x-UltraSharp",
    sub: "High-quality 4× upscaler",
    icon: "📈",
    sizeGb: 0.07,
    repoId: "lokCX/4x-Ultrasharp",
    targetDir: "upscalers",
    type: "other",
  },
  {
    id: "yolov8-face",
    name: "YOLOv8n Face",
    sub: "Face detector for Face Fix pass",
    icon: "👤",
    sizeGb: 0.01,
    repoId: "Bingsu/adetailer",
    targetDir: "face_detector",
    type: "other",
  },
  {
    id: "taesdxl",
    name: "TAESDXL",
    sub: "Tiny VAE for streaming latent previews",
    icon: "👁",
    sizeGb: 0.01,
    repoId: "madebyollin/taesdxl",
    targetDir: "taesdxl",
    type: "other",
  },
];

const essentialsModal = $<HTMLElement>("#essentials-modal");
const essentialsList = $<HTMLElement>("#essentials-list");
const essentialsCloseBtn = $<HTMLButtonElement>("#essentials-close-btn");
const essentialsInstallAllBtn = $<HTMLButtonElement>("#essentials-install-all");

interface EssentialState extends Essential {
  installed: boolean;
  downloading: boolean;
  done: boolean;
}

const essentialState: Record<string, EssentialState> = {};

async function checkEssentialsInstalled(): Promise<string[]> {
  const installed = await window.lavely.models.list();
  const installedNames = new Set(installed.map((m) => m.name.toLowerCase()));
  return ESSENTIALS.filter((e) => !installedNames.has(e.targetDir.toLowerCase())).map(
    (e) => e.id,
  );
}

function renderEssentialRow(e: EssentialState): HTMLElement {
  const row = document.createElement("div");
  row.className = "ess-row" + (e.installed || e.done ? " installed" : "");
  row.id = `ess-row-${e.id}`;

  const info = document.createElement("div");
  info.className = "ess-info";
  info.innerHTML = `<div class="ess-name">${e.name}</div><div class="ess-sub">${e.sub}</div>`;

  const iconEl = document.createElement("div");
  iconEl.className = "ess-icon";
  iconEl.textContent = e.icon;

  const sizeEl = document.createElement("div");
  sizeEl.className = "ess-size";
  sizeEl.textContent = `${e.sizeGb.toFixed(2)} GB`;

  const btn = document.createElement("button");
  btn.className = "btn btn-ghost btn-sm ess-btn";
  if (e.installed || e.done) {
    btn.textContent = "✓ Installed";
    btn.classList.add("done");
    btn.disabled = true;
  } else if (e.downloading) {
    btn.textContent = "Downloading…";
    btn.disabled = true;
  } else {
    btn.textContent = "⬇ Install";
    btn.addEventListener("click", () => installEssential(e.id));
  }

  row.appendChild(iconEl);
  row.appendChild(info);
  row.appendChild(sizeEl);
  row.appendChild(btn);
  return row;
}

function renderEssentials() {
  essentialsList.innerHTML = "";
  for (const e of ESSENTIALS) {
    const state = essentialState[e.id];
    if (!state) continue;
    essentialsList.appendChild(renderEssentialRow(state));
  }

  const allDone = Object.values(essentialState).every(
    (s) => s.installed || s.done,
  );
  essentialsInstallAllBtn.textContent = allDone ? "All set ✓" : "⬇ Install All";
  essentialsInstallAllBtn.disabled = allDone;
}

async function installEssential(id: string): Promise<void> {
  const state = essentialState[id];
  if (!state || state.installed || state.done || state.downloading) return;
  state.downloading = true;
  renderEssentials();

  return new Promise((resolve) => {
    const dispose = window.lavely.models.onDownloadEvent((msg: BridgeMsg) => {
      // We can't easily filter events per-download since the backend isn't
      // tagged, but since we install sequentially this is safe.
      if (msg.type === "done") {
        state.downloading = false;
        state.done = true;
        state.installed = true;
        renderEssentials();
        dispose();
        resolve();
      } else if (msg.type === "error") {
        state.downloading = false;
        renderEssentials();
        showToast(`Install failed: ${msg["message"]}`, "error", 6000);
        dispose();
        resolve();
      }
    });
    window.lavely.models.download(state.repoId, state.targetDir, state.type);
  });
}

async function installAllEssentials() {
  essentialsInstallAllBtn.disabled = true;
  for (const e of ESSENTIALS) {
    if (!essentialState[e.id]?.installed && !essentialState[e.id]?.done) {
      await installEssential(e.id);
    }
  }
  renderEssentials();
}

async function openEssentialsModal(force = false) {
  const missing = await checkEssentialsInstalled();
  if (missing.length === 0 && !force) return;
  for (const e of ESSENTIALS) {
    essentialState[e.id] = {
      ...e,
      installed: !missing.includes(e.id),
      downloading: false,
      done: false,
    };
  }
  renderEssentials();
  essentialsModal.classList.add("open");
}

essentialsCloseBtn.addEventListener("click", () => {
  essentialsModal.classList.remove("open");
});
essentialsInstallAllBtn.addEventListener("click", installAllEssentials);

// Run the check on first paint (but not if models are already in place)
setTimeout(() => {
  openEssentialsModal(false).catch(() => {
    /* ignore */
  });
}, 400);

// ─── Init ─────────────────────────────────────────────────────────────────────
activateScreen("chat");
