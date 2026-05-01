// My UI - Renderer process TypeScript
// Talks to main process through window.My (contextBridge)

import { marked } from "marked";

// Configure marked for safe, inline-friendly rendering
marked.setOptions({
  breaks: true, // Convert \n to <br>
  gfm: true, // GitHub-flavored markdown (tables, strikethrough, etc.)
});

/** Render markdown to HTML. Used for LLM chat bubbles and book recommendations. */
function renderMarkdown(md: string): string {
  return marked.parse(md) as string;
}

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
$("#btn-min").addEventListener("click", () => window.My.window.minimize());
$("#btn-max").addEventListener("click", () => window.My.window.maximize());
$("#btn-close").addEventListener("click", () => window.My.window.close());

// ─── App paths ────────────────────────────────────────────────────────────────
window.My.onPaths((paths) => {
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
const chatAttachBtn = $<HTMLButtonElement>("#chat-attach-btn");
const chatAttachPreview = $<HTMLDivElement>("#chat-attach-preview");
const chatAttachThumb = $<HTMLImageElement>("#chat-attach-thumb");
const chatAttachName = $<HTMLElement>("#chat-attach-name");
const chatAttachClearBtn = $<HTMLButtonElement>("#chat-attach-clear");
const chatVisionStatus = $<HTMLDivElement>("#chat-vision-status");
const chatVisionDot = $<HTMLSpanElement>("#chat-vision-dot");
const chatVisionText = $<HTMLSpanElement>("#chat-vision-text");
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
let chatAttachmentPath = "";
let visionLastCaptionChars = 0;
let visionLastModel = "";
let visionLastError = "";

type VisionUiState = "idle" | "ready" | "busy" | "error";

function updateVisionTooltip(statusText: string) {
  const parts: string[] = [statusText];
  if (visionLastModel) parts.push(`Model: ${visionLastModel}`);
  if (visionLastCaptionChars > 0) {
    parts.push(`Last caption length: ${visionLastCaptionChars} chars`);
  }
  if (visionLastError) parts.push(`Last error: ${visionLastError}`);
  chatVisionStatus.title = parts.join("\n");
}

function setVisionStatus(state: VisionUiState, message?: string) {
  chatVisionStatus.classList.remove("ready", "busy", "error");
  if (state !== "idle") chatVisionStatus.classList.add(state);

  if (state === "ready") {
    chatVisionDot.className = "dot on";
  } else if (state === "busy") {
    chatVisionDot.className = "dot loading";
  } else if (state === "error") {
    chatVisionDot.className = "dot";
    (chatVisionDot as HTMLElement).style.background = "var(--danger)";
  } else {
    chatVisionDot.className = "dot";
    (chatVisionDot as HTMLElement).style.background = "";
  }

  const statusText = message || "Vision: idle";
  chatVisionText.textContent = statusText;
  updateVisionTooltip(statusText);
}

setVisionStatus("idle", "Vision: idle");

function basenameFromPath(p: string): string {
  return p.replace(/\\/g, "/").split("/").pop() || p;
}

function setChatAttachment(path: string | null, preserveVisionStatus = false) {
  if (!path) {
    chatAttachmentPath = "";
    chatAttachPreview.classList.remove("visible");
    chatAttachThumb.src = "";
    chatAttachName.textContent = "";
    if (!preserveVisionStatus) setVisionStatus("idle", "Vision: idle");
    return;
  }
  chatAttachmentPath = path;
  chatAttachThumb.src = `file://${path}`;
  chatAttachName.textContent = basenameFromPath(path);
  chatAttachName.title = path;
  chatAttachPreview.classList.add("visible");
  setVisionStatus("idle", "Vision: image attached");
}

window.My.llm.onEvent((msg: BridgeMsg) => {
  if (msg.type === "status") {
    showToast(msg["message"] as string, "info");
  } else if (msg.type === "ready") {
    state.llmReady = true;
    state.llmLoading = false;
    updateLLMUI();
    showToast("LLM model ready ✓", "success");
  } else if (msg.type === "token") {
    // Ignore token streams from non-chat tools (Describe/Enhance/Books).
    if (!state.streaming && !currentAssistantBubble) return;
    if (!currentAssistantBubble) {
      currentAssistantText = "";
      const msgEl = appendMessage("assistant", "");
      currentAssistantBubble = msgEl.querySelector(".bubble")!;
      currentAssistantBubble.classList.add("typing-cursor");
    }
    currentAssistantText += msg.text ?? "";
    // Show raw text while streaming (fast, no reflow jitter)
    currentAssistantBubble.textContent = currentAssistantText;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else if (msg.type === "done") {
    // Ignore done events from non-chat tool requests.
    if (!state.streaming && !currentAssistantBubble) return;
    if (currentAssistantBubble) {
      currentAssistantBubble.classList.remove("typing-cursor");
      // Render final response as markdown
      currentAssistantBubble.innerHTML = renderMarkdown(currentAssistantText);
      // Save to conversation state
      state.messages.push({ role: "assistant", content: currentAssistantText });
      currentAssistantBubble = null;
      messagesEl.scrollTop = messagesEl.scrollHeight;
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
  const res = await window.My.llm.start();
  if (!res.ok) {
    state.llmLoading = false;
    updateLLMUI();
    showToast(`Failed: ${res.error}`, "error", 6000);
  }
});

llmStopBtn.addEventListener("click", async () => {
  await window.My.llm.stop();
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
  await window.My.llm.stop();
  state.llmReady = false;
  state.llmLoading = false;
  state.streaming = false;
  updateLLMUI();
  showToast("LLM unloaded", "info");
});

async function sendChat() {
  const text = chatInput.value.trim();
  if ((!text && !chatAttachmentPath) || !state.llmReady || state.streaming)
    return;

  chatInput.value = "";
  chatInput.style.height = "";

  let userText = text;
  if (chatAttachmentPath) {
    const fileName = basenameFromPath(chatAttachmentPath);
    setVisionStatus("busy", "Vision: analyzing image...");
    showToast("Analyzing attached image…", "info");
    const vision = await window.My.vision.describeImage(
      chatAttachmentPath,
      text || undefined,
    );
    if (vision.ok && vision.caption) {
      const caption = vision.caption.trim();
      visionLastCaptionChars = caption.length;
      visionLastModel = vision.model || "caption model";
      visionLastError = "";
      const usingFallback =
        !!vision.warning || visionLastModel === "basic-vision-fallback";
      if (usingFallback) {
        visionLastError = "Fallback mode active: basic visual analysis";
      }
      userText = `${text || "Please analyze the attached image."}\n\n[Attached image: ${fileName}]\n[Image description]\n${caption}`;
      const describeOut = document.querySelector<HTMLTextAreaElement>(
        "#pb-describe-output",
      );
      if (describeOut) describeOut.value = caption;
      setVisionStatus(
        "ready",
        usingFallback
          ? `Vision: basic mode (${visionLastCaptionChars} chars)`
          : `Vision: ready (${visionLastModel}, ${visionLastCaptionChars} chars)`,
      );
    } else {
      visionLastError = vision.error || "unknown error";
      userText = `${text || "Please analyze the attached image."}\n\n[Attached image: ${fileName}]`;
      setVisionStatus("error", "Vision: description failed");
      showToast(
        `Image description unavailable: ${visionLastError}`,
        "error",
        7000,
      );
    }
  }
  userText = userText.trim();

  appendMessage("user", userText);
  state.messages.push({ role: "user", content: userText });
  state.streaming = true;
  currentAssistantBubble = null;
  updateLLMUI();
  setChatAttachment(null, true);

  const request = {
    messages: state.messages,
    system: ($("#system-prompt") as HTMLTextAreaElement).value,
    max_tokens: parseInt(($("#max-tokens") as HTMLInputElement).value),
    temperature: parseFloat(($("#temperature") as HTMLInputElement).value),
    top_p: parseFloat(($("#top-p") as HTMLInputElement).value),
  };

  const res = await window.My.llm.chat(request);
  if (!res.ok) {
    state.streaming = false;
    updateLLMUI();
    showToast(`Chat error: ${res.error}`, "error");
  }
}

chatSendBtn.addEventListener("click", sendChat);

chatAttachBtn.addEventListener("click", async () => {
  const file = await window.My.dialog.openFile([
    { name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp"] },
  ]);
  if (!file) return;
  setChatAttachment(file);
});

chatAttachClearBtn.addEventListener("click", () => setChatAttachment(null));

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
  const position = ($("#pb-position") as HTMLTextAreaElement).value.trim();
  const weights = ($("#pb-weights") as HTMLTextAreaElement).value.trim();
  const positionWeight = parseFloat(
    ($("#pb-position-weight") as HTMLInputElement).value,
  );
  const weightedPosition = position
    ? positionWeight > 1
      ? `(${position}:${positionWeight.toFixed(2)})`
      : position
    : "";
  return [head, name, weightedPosition, weights].filter(Boolean).join(". ");
}

function updatePromptPreview() {
  const posWeightInput = $("#pb-position-weight") as HTMLInputElement;
  const posWeightVal = $("#pb-position-weight-val");
  if (posWeightInput && posWeightVal) {
    posWeightVal.textContent = parseFloat(posWeightInput.value).toFixed(2);
  }
  $("#full-prompt-preview").textContent = buildPrompt();
}

[
  "#pb-head",
  "#pb-name",
  "#pb-position",
  "#pb-weights",
  "#pb-position-weight",
].forEach((id) => {
  $(id).addEventListener("input", updatePromptPreview);
});
updatePromptPreview();

// ── Face Swap file picker ────────────────────────────────────────────────
let faceSwapSourcePath = "";

const faceSwapBrowseBtn = $<HTMLButtonElement>("#face-swap-browse");
const faceSwapFilename = $<HTMLElement>("#face-swap-filename");
const faceSwapPreview = $<HTMLImageElement>("#face-swap-preview");

faceSwapBrowseBtn.addEventListener("click", async () => {
  const file = await window.My.dialog.openFile([
    { name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] },
  ]);
  if (!file) return;
  faceSwapSourcePath = file;
  // Show just the filename
  const name = file.replace(/\\/g, "/").split("/").pop() || file;
  faceSwapFilename.textContent = name;
  faceSwapFilename.title = file;
  // Show a tiny circular preview
  faceSwapPreview.src = `file://${file}`;
  faceSwapPreview.style.display = "block";
});

// ── Position/pose presets ────────────────────────────────────────────────
const POSE_PRESETS = [
  {
    name: "crouching",
    prompt:
      "Down low with the body close to the ground, knees bent, usually heels up\u2014like you're ready to spring or sneak.",
  },
  {
    name: "squatting",
    prompt:
      "Knees bent deep, heels on the ground, body centered \u2014 classic squat position, grounded and balanced.",
  },
  {
    name: "low squat",
    prompt:
      "An even deeper squat, often with the hips almost touching the ground \u2014 legs open or closed depending on the mood.",
  },
  {
    name: "legs apart",
    prompt:
      "Standing or sitting with a visible gap between the legs\u2014confident, grounded, open stance.",
  },
  {
    name: "wide stance",
    prompt:
      "Feet planted far apart \u2014 strong, powerful, like taking up space on purpose.",
  },
  {
    name: "knees up",
    prompt:
      "One or both knees pulled upward \u2014 can be seated or lying down, adds energy and focus to the legs.",
  },
  {
    name: "one knee up",
    prompt:
      "One knee is raised, often while sitting or leaning\u2014adds dynamic lines and a casual tension.",
  },
  {
    name: "one knee on ground",
    prompt:
      "One leg down, one up \u2014 like a proposal pose or a lunge, adds asymmetry and balance.",
  },
  {
    name: "leaning forward",
    prompt:
      "Upper body tilts toward the camera or viewer \u2014 can feel inviting, intense, or intimate.",
  },
  {
    name: "hand on knee",
    prompt:
      "A hand resting on one or both knees \u2014 adds a touch of purpose or stability to the pose.",
  },
  {
    name: "hand between legs",
    prompt:
      "The hand is placed casually between the thighs \u2014 can be flirty, chill, or powerful depending on context.",
  },
  {
    name: "side profile squat",
    prompt:
      "Squatting but viewed from the side \u2014 shows off curves, posture, and shape.",
  },
  {
    name: "wall squat",
    prompt:
      "Back against the wall, knees bent in a squat \u2014 controlled, strong, and visually sharp.",
  },
  {
    name: "sitting on heels",
    prompt:
      "Sitting with butt next to the heels \u2014 neat, centered, and soft in posture.",
  },
  {
    name: "provocative pose",
    prompt:
      "A bold, attention-grabbing pose \u2014 body language turned all the way up.",
  },
  {
    name: "suggestive pose",
    prompt:
      "Hints without showing \u2014 body placement and angles do the talking.",
  },
  {
    name: "spreading legs",
    prompt:
      "Legs moved apart in a deliberate way \u2014 open, bold, and full of energy.",
  },
  {
    name: "legs up",
    prompt:
      "Legs are lifted \u2014 whether vertical, playful, or dramatic, it directs all eyes.",
  },
  {
    name: "arched back",
    prompt:
      "Spine curves inward, chest forward, hips back \u2014 adds drama, shape, and sensuality.",
  },
  {
    name: "looking back seductively",
    prompt:
      "The classic over-the-shoulder glance \u2014 flirty, mysterious, with just enough tease.",
  },
  {
    name: "resting on one knee",
    prompt:
      "More relaxed than a full kneel \u2014 balanced and easy, like a soft break in motion.",
  },
  {
    name: "bent over",
    prompt:
      "Torso leans forward, hips back \u2014 can be playful, strong, or risqu\u00e9 depending on angle.",
  },
  {
    name: "all fours",
    prompt:
      "Hands and knees on the ground \u2014 animalistic, grounded, and very body-focused.",
  },
  {
    name: "lying on stomach",
    prompt:
      "Body stretched out on the belly \u2014 soft lines, relaxed, or subtly inviting.",
  },
  {
    name: "lying on side",
    prompt:
      "Side-lying, one leg maybe bent \u2014 great for curve emphasis or a dreamy feel.",
  },
  {
    name: "thigh gap",
    prompt:
      "When the thighs don't touch \u2014 often highlighted when standing or posing with feet apart.",
  },
  {
    name: "legs together",
    prompt:
      "Legs placed closely \u2014 neat, elegant, or reserved, depending on the vibe.",
  },
  {
    name: "hips thrust forward",
    prompt:
      "The hips tilt or twist toward the front \u2014 showing off the curve, like a POW moment.",
  },
];

const posePresetSelect = $<HTMLSelectElement>("#pb-position-preset");
const poseAppendBtn = $<HTMLButtonElement>("#pb-position-append");
const poseUndoBtn = $<HTMLButtonElement>("#pb-position-undo");
const pbPosition = $<HTMLTextAreaElement>("#pb-position");
const positionAppendUndoStack: string[] = [];

function pushPositionUndoSnapshot() {
  positionAppendUndoStack.push(pbPosition.value);
  if (positionAppendUndoStack.length > 25) positionAppendUndoStack.shift();
  poseUndoBtn.disabled = positionAppendUndoStack.length === 0;
}

function appendToPosition(text: string) {
  const incoming = text.trim();
  if (!incoming) return;
  pushPositionUndoSnapshot();
  const current = pbPosition.value.trim();
  pbPosition.value = current
    ? `${current}${current.endsWith(",") ? "" : ","} ${incoming}`
    : incoming;
  updatePromptPreview();
}

// Populate the dropdown
POSE_PRESETS.forEach((p) => {
  const opt = document.createElement("option");
  opt.value = p.prompt;
  opt.textContent = p.name;
  posePresetSelect.appendChild(opt);
});

// Double-click on the dropdown → replace the textarea entirely
posePresetSelect.addEventListener("dblclick", () => {
  const val = posePresetSelect.value;
  if (!val) return;
  pbPosition.value = val;
  updatePromptPreview();
});

// "+ Add" button → append to the textarea (comma-separated)
poseAppendBtn.addEventListener("click", () => {
  const val = posePresetSelect.value;
  if (!val) {
    showToast("Pick a pose first", "info");
    return;
  }
  appendToPosition(val);
  // Reset dropdown so they can pick another
  posePresetSelect.value = "";
});

poseUndoBtn.addEventListener("click", () => {
  const prev = positionAppendUndoStack.pop();
  if (prev === undefined) return;
  pbPosition.value = prev;
  poseUndoBtn.disabled = positionAppendUndoStack.length === 0;
  updatePromptPreview();
  showToast("Reverted last append", "info");
});

// ── Enhance prompt via LLM ──────────────────────────────────────────────
const enhanceBtn = $<HTMLButtonElement>("#enhance-prompt-btn");
const describeBtn = $<HTMLButtonElement>("#describe-prompt-btn");
const describeModeFull = $<HTMLInputElement>("#describe-mode-full");
const describeModePosition = $<HTMLInputElement>("#describe-mode-position");
const describeModePositionAppend = $<HTMLInputElement>(
  "#describe-mode-position-append",
);
const pbDescribeOutput = $<HTMLTextAreaElement>("#pb-describe-output");
const pbDescribeUsePositionBtn = $<HTMLButtonElement>(
  "#pb-describe-use-position",
);
const pbDescribeUseWeightsBtn = $<HTMLButtonElement>(
  "#pb-describe-use-weights",
);

function setDescribeOutput(text: string) {
  pbDescribeOutput.value = text.trim();
}

pbDescribeUsePositionBtn.addEventListener("click", () => {
  const t = pbDescribeOutput.value.trim();
  if (!t) {
    showToast("Describe Output is empty", "info");
    return;
  }
  ($("#pb-position") as HTMLTextAreaElement).value = t;
  updatePromptPreview();
  showToast("Applied to Position", "success");
});

pbDescribeUseWeightsBtn.addEventListener("click", () => {
  const t = pbDescribeOutput.value.trim();
  if (!t) {
    showToast("Describe Output is empty", "info");
    return;
  }
  ($("#pb-weights") as HTMLTextAreaElement).value = t;
  updatePromptPreview();
  showToast("Applied to Weights", "success");
});

async function ensurePromptToolsLLMReady(): Promise<boolean> {
  // Fast path when UI state is already accurate.
  if (state.llmReady) return true;

  try {
    const status = await window.My.llm.status();
    state.llmReady = !!status.ready;
    state.llmLoading = !!status.running && !status.ready;
    updateLLMUI();

    if (status.ready) return true;
    if (status.running && !status.ready) {
      showToast("LLM is still loading. Please wait a moment.", "info");
      return false;
    }
  } catch {
    // Fall through to user-facing guidance.
  }

  showToast("Load the LLM model first (Chat panel → Load Model)", "error");
  return false;
}

describeBtn.addEventListener("click", async () => {
  if (!(await ensurePromptToolsLLMReady())) {
    return;
  }
  if (state.streaming) {
    showToast("LLM is busy. Wait for the current response to finish.", "info");
    return;
  }

  const subject = ($("#pb-name") as HTMLInputElement).value.trim();
  const position = ($("#pb-position") as HTMLTextAreaElement).value.trim();
  const weights = ($("#pb-weights") as HTMLTextAreaElement).value.trim();
  const describePositionOnly = describeModePosition.checked;
  const appendPosition = describeModePositionAppend.checked;

  if (!subject && !position && !weights) {
    showToast("Fill in at least one prompt field first", "info");
    return;
  }

  const raw = [subject, position, weights].filter(Boolean).join(". ");

  describeBtn.disabled = true;
  enhanceBtn.disabled = true;
  describeBtn.textContent = "Describing…";

  let result = "";
  let finished = false;

  const dispose = window.My.llm.onEvent((msg: BridgeMsg) => {
    if (finished) return;
    if (msg.type === "token") {
      result += (msg["text"] || "") as string;
    } else if (msg.type === "done") {
      finished = true;
      dispose();

      let cleaned = result.trim();
      cleaned = cleaned.replace(/^["'`]+|["'`]+$/g, "");
      setDescribeOutput(cleaned);

      if (describePositionOnly || appendPosition) {
        const posEl = $("#pb-position") as HTMLTextAreaElement;
        if (appendPosition) {
          appendToPosition(cleaned);
        } else {
          posEl.value = cleaned;
          updatePromptPreview();
        }
      } else {
        const parts = cleaned.split(
          /\.\s*(?=(?:soft|warm|dramatic|cinematic|natural|golden|studio|ambient|neon|moody|harsh|diffuse|rim|backlit|side|hard|subtle))/i,
        );
        if (parts.length >= 2) {
          ($("#pb-position") as HTMLTextAreaElement).value = parts[0].trim();
          ($("#pb-weights") as HTMLTextAreaElement).value = parts
            .slice(1)
            .join(". ")
            .trim();
        } else {
          ($("#pb-position") as HTMLTextAreaElement).value = cleaned;
        }
      }
      if (!describePositionOnly && !appendPosition) updatePromptPreview();
      describeBtn.disabled = false;
      enhanceBtn.disabled = false;
      describeBtn.textContent = "📝 Describe";
      showToast("Description applied ✓", "success");
    } else if (msg.type === "error") {
      finished = true;
      dispose();
      describeBtn.disabled = false;
      enhanceBtn.disabled = false;
      describeBtn.textContent = "📝 Describe";
      showToast(
        `Describe failed: ${msg["message"] || "unknown error"}`,
        "error",
        7000,
      );
    }
  });

  const systemPrompt =
    describePositionOnly || appendPosition
      ? "You are an expert visual pose describer for SDXL prompts. " +
        "Given rough user text, produce ONE concise position/action description only. " +
        "Output only plain text, no bullets, no quotes, no headings, under 45 words."
      : "You are an expert Stable Diffusion XL prompt engineer. " +
        "Given rough user text, write a concise visual description suitable for the Position/Action and style fields. " +
        "Output only plain text, no bullets, no quotes, under 90 words.";

  const describeReq = await window.My.llm.chat({
    messages: [{ role: "user", content: raw }],
    system: systemPrompt,
    max_tokens: describePositionOnly || appendPosition ? 140 : 220,
    temperature: 0.75,
    top_p: 0.9,
  });

  if (!describeReq.ok) {
    finished = true;
    dispose();
    describeBtn.disabled = false;
    enhanceBtn.disabled = false;
    describeBtn.textContent = "📝 Describe";
    showToast(`Describe failed: ${describeReq.error}`, "error", 6000);
    return;
  }

  setTimeout(() => {
    if (!finished) {
      finished = true;
      dispose();
      describeBtn.disabled = false;
      enhanceBtn.disabled = false;
      describeBtn.textContent = "📝 Describe";
      if (result.trim()) {
        const trimmed = result.trim();
        setDescribeOutput(trimmed);
        if (describePositionOnly || appendPosition) {
          const posEl = $("#pb-position") as HTMLTextAreaElement;
          const incoming = trimmed;
          if (appendPosition) {
            appendToPosition(incoming);
          } else {
            posEl.value = incoming;
            updatePromptPreview();
          }
        } else {
          ($("#pb-position") as HTMLTextAreaElement).value = result.trim();
          updatePromptPreview();
        }
        showToast("Partial description applied", "info");
      } else {
        showToast(
          "Describe timed out (model is taking too long). Try lower max tokens or click again.",
          "error",
          8000,
        );
      }
    }
  }, 180000);
});

enhanceBtn.addEventListener("click", async () => {
  if (!(await ensurePromptToolsLLMReady())) {
    return;
  }
  if (state.streaming) {
    showToast("LLM is busy. Wait for the current response to finish.", "info");
    return;
  }

  const subject = ($("#pb-name") as HTMLInputElement).value.trim();
  const position = ($("#pb-position") as HTMLTextAreaElement).value.trim();
  const weights = ($("#pb-weights") as HTMLTextAreaElement).value.trim();

  if (!subject && !position) {
    showToast("Fill in at least a subject or action first", "info");
    return;
  }

  const raw = [subject, position, weights].filter(Boolean).join(". ");

  enhanceBtn.disabled = true;
  enhanceBtn.textContent = "Enhancing…";

  // Collect the LLM response into a buffer.
  // We'll detach the listener once the response is done.
  let result = "";
  let finished = false;
  const dispose = window.My.llm.onEvent((msg: BridgeMsg) => {
    if (finished) return;
    if (msg.type === "token") {
      result += (msg["text"] || "") as string;
    } else if (msg.type === "done") {
      finished = true;
      dispose();

      // Parse the LLM output — it should be an improved prompt string.
      // Strip any wrapping quotes/backticks the LLM might add.
      let cleaned = result.trim();
      cleaned = cleaned.replace(/^["'`]+|["'`]+$/g, "");
      setDescribeOutput(cleaned);

      // Split into position/action + style/lighting if the LLM gave us
      // two clear sections, otherwise put everything into position.
      const parts = cleaned.split(
        /\.\s*(?=(?:soft|warm|dramatic|cinematic|natural|golden|studio|ambient|neon|moody|harsh|diffuse|rim|backlit|side|hard|subtle))/i,
      );
      if (parts.length >= 2) {
        ($("#pb-position") as HTMLTextAreaElement).value = parts[0].trim();
        ($("#pb-weights") as HTMLTextAreaElement).value = parts
          .slice(1)
          .join(". ")
          .trim();
      } else {
        ($("#pb-position") as HTMLTextAreaElement).value = cleaned;
      }
      updatePromptPreview();
      enhanceBtn.disabled = false;
      enhanceBtn.textContent = "✨ Enhance";
      showToast("Prompt enhanced ✓", "success");
    } else if (msg.type === "error") {
      finished = true;
      dispose();
      enhanceBtn.disabled = false;
      enhanceBtn.textContent = "✨ Enhance";
      showToast(
        `Enhance failed: ${msg["message"] || "unknown error"}`,
        "error",
        7000,
      );
    }
  });

  const systemPrompt =
    "You are an expert Stable Diffusion XL prompt engineer. " +
    "The user gives you a rough image idea. Rewrite it as a single, " +
    "detailed SDXL prompt that adds: specific visual details, body " +
    "language, facial expression, clothing/material textures, " +
    "composition, camera angle, and lighting. " +
    "Output ONLY the improved prompt — no explanation, no headings, " +
    "no bullet points, no quotes. Keep it under 120 words.";

  const enhanceReq = await window.My.llm.chat({
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: raw },
    ],
    max_tokens: 300,
    temperature: 0.8,
  });

  if (!enhanceReq.ok) {
    finished = true;
    dispose();
    enhanceBtn.disabled = false;
    enhanceBtn.textContent = "✨ Enhance";
    showToast(`Enhance failed: ${enhanceReq.error}`, "error", 6000);
    return;
  }

  // Timeout: if the LLM never fires "done" within 30s, give up
  setTimeout(() => {
    if (!finished) {
      finished = true;
      dispose();
      enhanceBtn.disabled = false;
      enhanceBtn.textContent = "✨ Enhance";
      if (result) {
        const trimmed = result.trim();
        setDescribeOutput(trimmed);
        ($("#pb-position") as HTMLTextAreaElement).value = trimmed;
        updatePromptPreview();
        showToast("Partial enhance applied", "info");
      } else {
        showToast(
          "Enhance timed out (model is taking too long). Try lower max tokens or click again.",
          "error",
          8000,
        );
      }
    }
  }, 180000);
});

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
window.My.image.onEvent((msg: BridgeMsg) => {
  if (msg.type === "status") {
    const text = msg["message"] as string;
    // While generating, show status in the progress bar for clear phase feedback
    if (state.generating) {
      progressLabelText.textContent = text;
      progressEta.textContent = "";
    } else {
      showToast(text, "info");
    }
  } else if (msg.type === "log" && state.generating) {
    // Face detailer progress etc. → inline in the progress bar
    const text = String(msg["text"] || "");
    if (text) progressLabelText.textContent = text;
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
      window.My.history.save({
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
  const res = await window.My.image.start();
  if (!res.ok) {
    state.imageLoading = false;
    updateImageUI();
    showToast(`Failed: ${res.error}`, "error", 6000);
  }
});

imgStopBtn.addEventListener("click", async () => {
  await window.My.image.stop();
  state.imageReady = false;
  state.imageLoading = false;
  state.generating = false;
  progressWrap.style.display = "none";
  progressFill.style.width = "0%";
  updateImageUI();
  showToast("Image process stopped", "info");
});

imgUnloadBtn.addEventListener("click", async () => {
  await window.My.image.stop();
  state.imageReady = false;
  state.imageLoading = false;
  state.generating = false;
  progressWrap.style.display = "none";
  progressFill.style.width = "0%";
  updateImageUI();
  showToast("Image model unloaded", "info");
});

generateStopBtn.addEventListener("click", async () => {
  await window.My.image.stop();
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

  const doFaceSwap = ($("#img-face-swap") as HTMLInputElement).checked;
  const request: Record<string, unknown> = {
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
    face_swap: doFaceSwap,
  };
  if (doFaceSwap && faceSwapSourcePath) {
    request.face_swap_source = faceSwapSourcePath;
  } else if (doFaceSwap && !faceSwapSourcePath) {
    showToast("Select a face image first (Choose face...)", "error");
    state.generating = false;
    updateImageUI();
    progressWrap.style.display = "none";
    return;
  }

  const res = await window.My.image.generate(request);
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

let mediaCurrentDir = ""; // "" = outputs root
let mediaRenderToken = 0;
const MEDIA_RENDER_BATCH_SIZE = 48;
const MEDIA_PLACEHOLDER_DATA_URL =
  "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA=";
const thumbnailRequestCache = new Map<string, Promise<string>>();

let mediaThumbObserver: IntersectionObserver | null = null;

function ensureMediaThumbObserver() {
  if (mediaThumbObserver) return mediaThumbObserver;
  mediaThumbObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const img = entry.target as HTMLImageElement;
        const fullPath = img.dataset.fullPath;
        if (!fullPath) return;
        mediaThumbObserver?.unobserve(img);
        void loadMediaThumbnail(img, fullPath);
      });
    },
    { rootMargin: "600px 0px" },
  );
  return mediaThumbObserver;
}

async function loadMediaThumbnail(img: HTMLImageElement, fullPath: string) {
  try {
    let pending = thumbnailRequestCache.get(fullPath);
    if (!pending) {
      pending = window.My.media
        .getThumbnail(fullPath, 360)
        .then((res) => {
          if (res.ok && res.path) return `file://${res.path}`;
          return `file://${fullPath}`;
        })
        .catch(() => `file://${fullPath}`);
      thumbnailRequestCache.set(fullPath, pending);
    }
    img.src = await pending;
  } catch {
    img.src = `file://${fullPath}`;
  }
}

function renderMediaBreadcrumb() {
  const bc = $("#media-breadcrumb");
  bc.innerHTML = "";
  const parts = ["outputs", ...mediaCurrentDir.split("/").filter(Boolean)];

  parts.forEach((part, i) => {
    if (i > 0) {
      const sep = document.createElement("span");
      sep.className = "sep";
      sep.textContent = "›";
      bc.appendChild(sep);
    }
    const el = document.createElement("span");
    el.textContent = part;
    if (i === parts.length - 1) {
      el.className = "current";
    } else {
      // Clicking a breadcrumb navigates up
      const targetDir = i === 0 ? "" : parts.slice(1, i + 1).join("/");
      el.addEventListener("click", () => {
        mediaCurrentDir = targetDir;
        loadMediaGrid();
      });
    }
    bc.appendChild(el);
  });
}

async function loadMediaGrid() {
  const renderToken = ++mediaRenderToken;
  const grid = $("#media-grid");
  grid.innerHTML =
    '<div class="empty-state"><div class="es-icon">⏳</div><p>Loading…</p></div>';
  renderMediaBreadcrumb();

  const listing = await window.My.media.list(mediaCurrentDir || undefined);
  if (renderToken !== mediaRenderToken) return;
  const { folders, files } = listing;

  if (!folders.length && !files.length) {
    grid.innerHTML =
      '<div class="empty-state"><div class="es-icon">🖼️</div><p>No images yet. Generate something!</p></div>';
    return;
  }
  grid.innerHTML = "";

  // Render folders first
  folders.forEach((folder) => {
    const card = document.createElement("div");
    card.className = "folder-card";

    const icon = document.createElement("div");
    icon.className = "folder-icon";
    icon.textContent = "📁";

    const name = document.createElement("div");
    name.className = "folder-name";
    name.textContent = folder.name;

    const del = document.createElement("button");
    del.className = "folder-del";
    del.textContent = "🗑";
    del.title = "Delete folder";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      const ok = await askConfirm(
        "Delete folder?",
        `"${folder.name}" and all its contents will be permanently deleted.`,
      );
      if (!ok) return;
      const res = await window.My.media.deleteFolder(folder.rel);
      if (res.ok) {
        card.remove();
        showToast(`Folder "${folder.name}" deleted`, "success");
      } else {
        showToast(`Error: ${res.error}`, "error");
      }
    });

    card.addEventListener("click", () => {
      mediaCurrentDir = folder.rel;
      loadMediaGrid();
    });

    card.appendChild(icon);
    card.appendChild(name);
    card.appendChild(del);
    grid.appendChild(card);
  });

  // Collect folder names for the "move to" menu
  const folderNames = folders.map((f) => f.rel);

  // Render image files in batches so large folders stay responsive.
  for (let i = 0; i < files.length; i += MEDIA_RENDER_BATCH_SIZE) {
    if (renderToken !== mediaRenderToken) return;
    const batch = files.slice(i, i + MEDIA_RENDER_BATCH_SIZE);
    const frag = document.createDocumentFragment();
    batch.forEach((f) => {
      const card = document.createElement("div");
      card.className = "media-card";
      card.innerHTML = `
      <img src="${MEDIA_PLACEHOLDER_DATA_URL}" loading="lazy" decoding="async" alt="${f.name}" />
      <div class="media-card-info">
        <div class="media-card-name" title="${f.name}">${f.name}</div>
      </div>
      <div class="media-card-actions">
        <button title="Move to folder" class="mc-move">📂→</button>
        <button title="Open" class="mc-open">↗</button>
        <button title="Delete" class="mc-del">🗑</button>
      </div>
    `;
      const imgEl = card.querySelector("img") as HTMLImageElement;
      imgEl.dataset.fullPath = f.path;
      ensureMediaThumbObserver().observe(imgEl);
      imgEl.addEventListener("click", () =>
        openLightbox(`file://${f.path}`, f.path),
      );
      card.querySelector(".mc-open")!.addEventListener("click", (e) => {
        e.stopPropagation();
        window.My.media.open(f.path);
      });
      card.querySelector(".mc-move")!.addEventListener("click", async (e) => {
        e.stopPropagation();
        // Build a list of available destinations: parent ("outputs root") + sibling folders
        const destinations = ["(root)"];
        if (mediaCurrentDir) destinations.push("(root)");
        folderNames.forEach((fn) => destinations.push(fn));

        const dest = await askInput(
          "Move to folder",
          `Move "${f.name}" to which folder? Type a folder name (existing or new).`,
          "e.g. favorites",
          folderNames[0] || "",
        );
        if (!dest) return;
        const targetFolder = dest === "(root)" ? "" : dest;
        const res = await window.My.media.move(f.path, targetFolder);
        if (res.ok) {
          card.remove();
          showToast(`Moved to ${targetFolder || "root"}`, "success");
        } else {
          showToast(`Move failed: ${res.error}`, "error");
        }
      });
      card.querySelector(".mc-del")!.addEventListener("click", async (e) => {
        e.stopPropagation();
        const ok = await askConfirm("Delete image?", f.name);
        if (!ok) return;
        const res = await window.My.media.delete(f.path);
        if (res.ok) {
          card.remove();
          showToast("Deleted", "success");
        } else showToast(`Error: ${res.error}`, "error");
      });
      frag.appendChild(card);
    });
    grid.appendChild(frag);
    // Yield to the browser so input/scroll stays smooth while rendering many cards.
    await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
  }
}

$("#media-refresh-btn").addEventListener("click", loadMediaGrid);

$("#media-new-folder-btn").addEventListener("click", async () => {
  const name = await askInput(
    "New folder",
    "Create a subfolder inside the current directory.",
    "e.g. favorites",
  );
  if (!name) return;
  const fullName = mediaCurrentDir ? `${mediaCurrentDir}/${name}` : name;
  const res = await window.My.media.createFolder(fullName);
  if (res.ok) {
    showToast(`Folder "${name}" created`, "success");
    loadMediaGrid();
  } else {
    showToast(`Error: ${res.error}`, "error");
  }
});

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
  $("#lb-open").addEventListener("click", () => window.My.media.open(filePath));
  $("#lb-folder").addEventListener("click", () =>
    window.My.media.show(filePath),
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
  adapter: "🧬",
  unknown: "📦",
};

async function loadModelsList() {
  const list = $("#models-list");
  list.innerHTML = "";
  const models = await window.My.models.list();
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

window.My.models.onDownloadEvent((msg: BridgeMsg) => {
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
  const dir = await window.My.dialog.openDir();
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
    const info = await window.My.system.gpuInfo();
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

    const tags = [
      `<span class="cc-tag cat-${e.category}">${CAT_LABEL[e.category]}</span>`,
    ];
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
        e.category === "llm"
          ? "llm"
          : e.category === "image"
            ? "image"
            : "other";
      logDownload(
        `▶ Installing ${e.name} (${e.repoId}) → models/${e.targetDir}`,
      );
      await window.My.models.download(e.repoId, e.targetDir, type);
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
  catalogState.entries = await window.My.catalog.list(
    catalogState.vramGb || undefined,
  );
  // Re-fetch with no filter if "show all" — the filtering is done client-side
  // so we can toggle instantly without a round-trip.
  if (catalogShowAll.checked) {
    catalogState.entries = await window.My.catalog.list();
  }
  renderCatalog();
}

catalogShowAll.addEventListener("change", async () => {
  // When toggling to "show all", fetch the full list
  catalogState.entries = await window.My.catalog.list(
    catalogShowAll.checked ? undefined : catalogState.vramGb || undefined,
  );
  renderCatalog();
});
catalogCategoryFilter.addEventListener("change", renderCatalog);

// Refresh catalog state whenever a download completes (installed flag flips)
window.My.models.onDownloadEvent((msg: BridgeMsg) => {
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
  await window.My.models.download(repoId, targetDir, type);
});

$("#run-diagnostics-btn").addEventListener("click", async () => {
  diagnosticsLog.textContent = "";
  logDiagnostics("Running environment diagnostics...");
  const info = await window.My.system.diagnostics();
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
const gpuClearThumbCacheBtn = $<HTMLButtonElement>(
  "#gpu-clear-thumb-cache-btn",
);

async function loadGpuInfo() {
  gpuContent.innerHTML =
    '<div class="empty-state"><div class="es-icon">⏳</div><p>Loading GPU information…</p></div>';
  gpuRaw.textContent = "";

  const data = await window.My.system.gpuInfo();

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

gpuClearThumbCacheBtn.addEventListener("click", async () => {
  gpuClearThumbCacheBtn.disabled = true;
  try {
    const res = await window.My.system.clearThumbnailCache();
    if (!res.ok) {
      showToast(
        `Failed to clear thumbnail cache: ${res.error ?? "unknown"}`,
        "error",
        5000,
      );
      return;
    }

    thumbnailRequestCache.clear();

    const mb = ((res.removedBytes ?? 0) / (1024 * 1024)).toFixed(2);
    showToast(
      `Thumbnail cache cleared (${res.removedFiles ?? 0} files, ${mb} MB)`,
      "success",
    );
    gpuRaw.textContent = JSON.stringify(
      {
        thumbnailCache: {
          cleared: true,
          removedFiles: res.removedFiles ?? 0,
          removedBytes: res.removedBytes ?? 0,
        },
      },
      null,
      2,
    );
  } finally {
    gpuClearThumbCacheBtn.disabled = false;
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// BENCHMARK (lives on the GPU screen)
// ─────────────────────────────────────────────────────────────────────────────

const benchState = {
  running: false,
  totalTrials: 0,
  doneTrials: 0,
};

const benchLog = $<HTMLElement>("#bench-log");
const benchBar = $<HTMLElement>("#bench-bar");
const benchStartBtn = $<HTMLButtonElement>("#bench-start-btn");
const benchStopBtn = $<HTMLButtonElement>("#bench-stop-btn");
const benchClearLogBtn = $<HTMLButtonElement>("#bench-clear-log-btn");
const benchStatusBadge = $<HTMLElement>("#bench-status-badge");
const benchRefreshResultsBtn = $<HTMLButtonElement>(
  "#bench-refresh-results-btn",
);
const benchOpenResultsDirBtn = $<HTMLButtonElement>(
  "#bench-open-results-dir-btn",
);
const benchResultsList = $<HTMLElement>("#bench-results-list");
const benchResultsDirBadge = $<HTMLElement>("#bench-results-dir-badge");
const benchResultDetail = $<HTMLElement>("#bench-result-detail");

function benchLogLine(text: string, cls: "" | "ok" | "err" = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  benchLog.appendChild(line);
  benchLog.scrollTop = benchLog.scrollHeight;
}

function setBenchRunning(running: boolean) {
  benchState.running = running;
  benchStartBtn.disabled = running;
  benchStopBtn.disabled = !running;
  benchStartBtn.textContent = running ? "Running…" : "▶ Start Benchmark";
  benchStatusBadge.textContent = running ? "Running…" : "Idle";
}

function readBenchConfig(): BenchmarkConfig {
  const v = (id: string) => ($(`#${id}`) as HTMLInputElement).value.trim();
  const trialsRaw = parseInt(v("bench-trials"), 10);
  return {
    models: v("bench-models"),
    tasks: v("bench-tasks"),
    conditions: v("bench-conditions") || "raw,autotune",
    trials: Number.isFinite(trialsRaw) && trialsRaw > 0 ? trialsRaw : 2,
    output: v("bench-output") || undefined,
    dryRun: ($("#bench-dry-run") as HTMLInputElement).checked,
  };
}

function estimateTotalTrials(cfg: BenchmarkConfig): number {
  const count = (s: string) =>
    s
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean).length;
  const models = Math.max(1, count(cfg.models));
  // tasks blank => "all" — script currently has 2 tasks (code_debugger, research_synth)
  const tasks = cfg.tasks ? Math.max(1, count(cfg.tasks)) : 2;
  const conditions = Math.max(1, count(cfg.conditions));
  return models * tasks * conditions * Math.max(1, cfg.trials);
}

benchStartBtn.addEventListener("click", async () => {
  const cfg = readBenchConfig();
  if (!cfg.models) {
    showToast("Please specify at least one model", "error");
    return;
  }
  benchLog.innerHTML = "";
  benchBar.style.width = "0%";
  benchState.totalTrials = estimateTotalTrials(cfg);
  benchState.doneTrials = 0;
  benchLogLine(`Starting benchmark with config:`);
  benchLogLine(JSON.stringify(cfg, null, 2));
  benchLogLine(`Estimated trials: ${benchState.totalTrials}`);

  setBenchRunning(true);
  const res = await window.My.bench.start(cfg);
  if (!res.ok) {
    benchLogLine(`✗ Failed to start: ${res.error}`, "err");
    showToast(`Benchmark failed to start: ${res.error}`, "error", 6000);
    setBenchRunning(false);
  }
});

benchStopBtn.addEventListener("click", async () => {
  await window.My.bench.stop();
  benchLogLine("■ Benchmark stopped by user", "err");
  setBenchRunning(false);
});

benchClearLogBtn.addEventListener("click", () => {
  benchLog.innerHTML = "";
});

window.My.bench.onEvent((msg: BridgeMsg) => {
  switch (msg.type) {
    case "status":
      benchLogLine(`ℹ ${msg["message"]}`);
      break;
    case "log": {
      const text = String(msg["text"] ?? "");
      benchLogLine(text);
      // Heuristic: count completed trials based on the script's
      // "✓ ... turns, ...s" / "✗ ..." per-trial output line.
      if (
        /^\s*[✓✗]\s+\d+\s+turns/.test(text) ||
        /\b\d+\s+turns,\s/.test(text)
      ) {
        benchState.doneTrials += 1;
        if (benchState.totalTrials > 0) {
          const pct = Math.min(
            100,
            (benchState.doneTrials / benchState.totalTrials) * 100,
          );
          benchBar.style.width = `${pct.toFixed(1)}%`;
        }
      }
      if (/^Results written to /.test(text)) {
        // Refresh history when results land.
        refreshBenchResults();
      }
      break;
    }
    case "stderr": {
      const t = String(msg["text"] ?? "");
      if (
        t &&
        !t.includes("FutureWarning") &&
        !t.includes("UserWarning") &&
        !t.includes("warnings.warn")
      ) {
        benchLogLine(t);
      }
      break;
    }
    case "error":
      benchLogLine(`✗ ${msg["message"]}`, "err");
      showToast(`Benchmark error: ${msg["message"]}`, "error", 6000);
      setBenchRunning(false);
      break;
    case "exit":
      if (msg["code"] === 0) {
        benchLogLine(`✓ Benchmark complete (exit 0)`, "ok");
        benchBar.style.width = "100%";
        showToast("Benchmark complete ✓", "success");
        refreshBenchResults();
      } else {
        benchLogLine(`Process exited with code ${msg["code"]}`, "err");
      }
      setBenchRunning(false);
      break;
  }
});

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(ms: number): string {
  try {
    return new Date(ms).toLocaleString();
  } catch {
    return String(ms);
  }
}

async function refreshBenchResults() {
  const res = await window.My.bench.listResults();
  benchResultsDirBadge.textContent = res.dir;
  benchResultsDirBadge.title = res.dir;
  if (!res.ok) {
    benchResultsList.innerHTML = `<div class="empty-state" style="padding: 12px"><p>Failed to list results: ${res.error ?? "unknown error"}</p></div>`;
    return;
  }
  if (!res.files.length) {
    benchResultsList.innerHTML = `<div class="empty-state" style="padding: 12px"><p>No benchmarks yet. Run one to see results here.</p></div>`;
    return;
  }
  benchResultsList.innerHTML = "";
  for (const f of res.files) {
    const row = document.createElement("div");
    row.className = "bench-result-row";
    row.innerHTML = `
      <div>
        <div>${f.name}</div>
        <div class="br-meta">${formatDate(f.mtime)} · ${formatBytes(f.size)}</div>
      </div>
      <div class="br-actions">
        <button class="btn btn-ghost btn-sm" data-act="view">View</button>
      </div>
    `;
    const viewBtn = row.querySelector<HTMLButtonElement>('[data-act="view"]')!;
    viewBtn.addEventListener("click", () => viewBenchResult(f.path));
    benchResultsList.appendChild(row);
  }
}

async function viewBenchResult(filePath: string) {
  const res = await window.My.bench.getResult(filePath);
  if (!res.ok) {
    showToast(`Failed to read result: ${res.error}`, "error", 6000);
    return;
  }
  benchResultDetail.textContent = JSON.stringify(res.data, null, 2);
  benchResultDetail.classList.add("visible");
  benchResultDetail.scrollTop = 0;
  renderBenchCharts(res.data);
}

interface BenchSummary {
  task: string;
  model: string;
  condition: string;
  n_trials: number;
  avg_ttft_ms: number;
  ttft_slope_ms_per_turn: number;
  avg_model_reloads: number;
  avg_wall_s: number;
  avg_peak_ram_gb: number;
  total_swap_events: number;
  tool_call_errors: number;
  success_rate: number;
  avg_context_tokens_final: number;
}
interface BenchTrial {
  task_success: boolean;
  error?: string;
}
interface BenchData {
  meta?: { models?: string[]; tasks?: string[]; conditions?: string[] };
  trials?: BenchTrial[];
  summaries?: BenchSummary[];
}

const benchChartsEl = $<HTMLElement>("#bench-charts");

function renderBenchCharts(data: BenchData) {
  benchChartsEl.innerHTML = "";
  benchChartsEl.classList.add("visible");

  const summaries = data.summaries ?? [];
  if (!summaries.length) {
    benchChartsEl.innerHTML = `<div class="bench-chart"><h4>Charts</h4><div class="bc-empty">No summaries in this result file.</div></div>`;
    return;
  }

  const allErrored = (data.trials ?? []).every((t) => !t.task_success);
  const errBanner = allErrored
    ? `<div class="bc-empty" style="color: var(--danger); margin-bottom: 6px;">All trials failed — metrics shown are from error paths only.</div>`
    : "";

  benchChartsEl.appendChild(
    chartBlock(
      "Avg wall time (s) by task × condition",
      barChart(
        summaries.map((s) => ({
          label: `${s.task}\n${s.condition}`,
          value: s.avg_wall_s,
          ok: s.success_rate > 0,
        })),
        { unit: "s", decimals: 2 },
      ),
      errBanner,
    ),
  );

  benchChartsEl.appendChild(
    chartBlock(
      "Success rate by condition",
      barChart(
        summaries.map((s) => ({
          label: `${s.task}\n${s.condition}`,
          value: s.success_rate * 100,
          ok: s.success_rate > 0,
        })),
        { unit: "%", decimals: 0, max: 100 },
      ),
    ),
  );

  benchChartsEl.appendChild(
    chartBlock(
      "Avg TTFT (ms) by condition",
      barChart(
        summaries.map((s) => ({
          label: `${s.task}\n${s.condition}`,
          value: s.avg_ttft_ms,
          ok: s.avg_ttft_ms > 0,
        })),
        { unit: "ms", decimals: 0 },
      ),
    ),
  );

  benchChartsEl.appendChild(
    chartBlock(
      "Final context tokens",
      barChart(
        summaries.map((s) => ({
          label: `${s.task}\n${s.condition}`,
          value: s.avg_context_tokens_final,
          ok: true,
        })),
        { unit: "tok", decimals: 0 },
      ),
    ),
  );
}

function chartBlock(
  title: string,
  svg: string,
  banner: string = "",
): HTMLElement {
  const div = document.createElement("div");
  div.className = "bench-chart";
  div.innerHTML = `<h4>${title}</h4>${banner}${svg}`;
  return div;
}

function barChart(
  rows: Array<{ label: string; value: number; ok: boolean }>,
  opts: { unit?: string; decimals?: number; max?: number } = {},
): string {
  if (!rows.length) return `<div class="bc-empty">no data</div>`;
  const W = 320;
  const H = 180;
  const padL = 28;
  const padR = 8;
  const padT = 10;
  const padB = 36;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const dataMax = Math.max(...rows.map((r) => r.value), 0);
  const max = opts.max ?? (dataMax === 0 ? 1 : dataMax * 1.15);
  const barW = innerW / rows.length;
  const dec = opts.decimals ?? 1;
  const unit = opts.unit ?? "";

  const bars = rows
    .map((r, i) => {
      const h = max > 0 ? (r.value / max) * innerH : 0;
      const x = padL + i * barW + barW * 0.15;
      const y = padT + innerH - h;
      const w = barW * 0.7;
      const cls = r.ok ? "bc-bar" : "bc-bar-fail";
      const lines = r.label.split("\n");
      const labelTspans = lines
        .map(
          (l, j) =>
            `<tspan x="${x + w / 2}" dy="${j === 0 ? "1.1em" : "1em"}">${escapeXml(l)}</tspan>`,
        )
        .join("");
      const valStr = `${r.value.toFixed(dec)}${unit}`;
      return `
        <rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${Math.max(h, 0)}" rx="2" />
        <text class="bc-value" x="${x + w / 2}" y="${y - 3}" text-anchor="middle">${valStr}</text>
        <text class="bc-label" x="${x + w / 2}" y="${padT + innerH + 4}" text-anchor="middle">${labelTspans}</text>
      `;
    })
    .join("");

  const yTick = `${max.toFixed(dec)}${unit}`;
  return `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
      <line class="bc-axis" x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + innerH}" />
      <line class="bc-axis" x1="${padL}" y1="${padT + innerH}" x2="${W - padR}" y2="${padT + innerH}" />
      <text class="bc-tick" x="${padL - 4}" y="${padT + 3}" text-anchor="end">${yTick}</text>
      <text class="bc-tick" x="${padL - 4}" y="${padT + innerH + 3}" text-anchor="end">0</text>
      ${bars}
    </svg>
  `;
}

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

benchRefreshResultsBtn.addEventListener("click", refreshBenchResults);
benchOpenResultsDirBtn.addEventListener("click", async () => {
  await window.My.bench.openResultsDir();
});

// Refresh benchmark history when the GPU tab is opened.
document
  .querySelectorAll<HTMLElement>('.nav-btn[data-screen="gpu"]')
  .forEach((btn) => {
    btn.addEventListener("click", () => {
      refreshBenchResults();
    });
  });

// Initial load (so the list is populated even if user opens the screen directly).
refreshBenchResults();

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
const trainTestBtn = $<HTMLButtonElement>("#train-test-btn");
const trainMStep = $("#train-m-step");
const trainMEpoch = $("#train-m-epoch");
const trainMLoss = $("#train-m-loss");
const trainMLr = $("#train-m-lr");
let trainTestBusy = false;

function trainLogLine(
  text: string,
  cls: "" | "ok" | "err" = "",
  kind: "sys" | "info" | "cmd" | "warn" = "sys",
) {
  const line = document.createElement("div");
  line.className = `line k-${kind}`;
  if (cls) line.classList.add(cls);

  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = new Date().toLocaleTimeString([], {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const prompt = document.createElement("span");
  prompt.className = "prompt";
  prompt.textContent = cls === "err" ? "!" : cls === "ok" ? "✓" : "$";

  const msg = document.createElement("span");
  msg.className = "msg";
  msg.textContent = text;

  line.appendChild(ts);
  line.appendChild(prompt);
  line.appendChild(msg);
  trainLog.appendChild(line);
  trainLog.scrollTop = trainLog.scrollHeight;
}

function trainLogRaw(rawText: string) {
  const lines = String(rawText)
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  for (const line of lines) {
    // Some bridge steps emit JSONL; decode and render friendly log lines.
    if (line.startsWith("{") && line.endsWith("}")) {
      try {
        const parsed = JSON.parse(line);
        const type = String(parsed.type || "");
        if (type === "status" && parsed.message) {
          trainLogLine(String(parsed.message), "", "info");
          continue;
        }
        if (type === "done" && parsed.output) {
          trainLogLine(`Merged model saved to ${parsed.output}`, "ok");
          continue;
        }
      } catch {
        // non-JSON line, fall through
      }
    }
    trainLogLine(line);
  }
}

function setTrainTestBusy(busy: boolean) {
  trainTestBusy = busy;
  trainTestBtn.disabled = busy || trainState.running;
  trainTestBtn.textContent = busy ? "⏳ Testing…" : "▶ Test in Chat";
}

async function maybeEnableTrainTestButton() {
  const cfg = readTrainConfig();
  const mergedPath = `${cfg.output_dir}-merged`;
  const hasMerged = await window.My.models.exists(mergedPath);
  trainTestBtn.disabled = trainState.running || trainTestBusy || !hasMerged;
  if (hasMerged) {
    trainTestBtn.dataset["mergedPath"] = mergedPath;
  }
}

function setTrainRunning(running: boolean) {
  trainState.running = running;
  trainStartBtn.disabled = running;
  trainStopBtn.disabled = !running;
  trainTestBtn.disabled = running || trainTestBusy;
  trainStartBtn.textContent = running ? "Training…" : "▶ Start Training";
}

async function runMergeFromTrain(cfg: TrainConfig) {
  if (!cfg.model_path || !cfg.output_dir) {
    return {
      ok: false,
      error: "Set base model + adapter output directory first",
      mergedPath: "",
    };
  }
  const mergedPath = cfg.output_dir + "-merged";
  trainLogLine(`merge ${cfg.output_dir} -> ${mergedPath}`, "", "cmd");
  const res = await window.My.train.merge(
    cfg.model_path,
    cfg.output_dir,
    mergedPath,
  );
  if (res.ok) {
    trainLogLine(`Merged model saved to ${mergedPath}`, "ok");
    trainTestBtn.dataset["mergedPath"] = mergedPath;
    return { ok: true, mergedPath };
  }
  return { ok: false, error: res.error || "Merge failed", mergedPath };
}

// Browse handlers — directory for model/output, file for dataset
$("#train-model-browse").addEventListener("click", async () => {
  const dir = await window.My.dialog.openDir();
  if (dir) ($("#train-model-path") as HTMLInputElement).value = dir;
});
$("#train-output-browse").addEventListener("click", async () => {
  const dir = await window.My.dialog.openDir();
  if (dir) {
    ($("#train-output-path") as HTMLInputElement).value = dir;
    void maybeEnableTrainTestButton();
  }
});
$("#train-dataset-browse").addEventListener("click", async () => {
  const file = await window.My.dialog.openFile([
    { name: "Dataset", extensions: ["jsonl", "json"] },
    { name: "All Files", extensions: ["*"] },
  ]);
  if (file) ($("#train-dataset-path") as HTMLInputElement).value = file;
});

// Prefill model path from app paths
window.My.onPaths((paths) => {
  const modelInput = $<HTMLInputElement>("#train-model-path");
  if (!modelInput.value && paths.llmModel) modelInput.value = paths.llmModel;
  void maybeEnableTrainTestButton();
});

function readTrainConfig(): TrainConfig {
  const v = (id: string) => ($(`#${id}`) as HTMLInputElement).value;
  const n = (id: string) => parseFloat(v(id));
  const i = (id: string) => parseInt(v(id), 10);
  return {
    model_path: v("train-model-path").trim(),
    dataset_path: v("train-dataset-path").trim(),
    output_dir: v("train-output-path").trim() || "models/My-lm-lora",
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
  const res = await window.My.train.start(cfg);
  if (!res.ok) {
    trainLogLine(`✗ Failed to start: ${res.error}`, "err");
    showToast(`Training failed to start: ${res.error}`, "error", 6000);
    setTrainRunning(false);
  }
});

trainStopBtn.addEventListener("click", async () => {
  await window.My.train.stop();
  trainLogLine("■ Training stopped by user", "err");
  setTrainRunning(false);
});

trainMergeBtn.addEventListener("click", async () => {
  const cfg = readTrainConfig();
  const res = await runMergeFromTrain(cfg);
  if (res.ok) {
    showToast("Merge complete ✓", "success");
    await maybeEnableTrainTestButton();
  } else {
    trainLogLine(`Merge failed: ${res.error}`, "err");
    showToast(`Merge failed: ${res.error}`, "error", 6000);
  }
});

trainTestBtn.addEventListener("click", async () => {
  const cfg = readTrainConfig();
  if (!cfg.output_dir) {
    showToast("Set adapter output directory first", "error");
    return;
  }

  setTrainTestBusy(true);
  try {
    const mergedPath = `${cfg.output_dir}-merged`;
    let canLoadMerged = await window.My.models.exists(mergedPath);

    // If no merged model exists yet, merge now (requires base model + adapter dir).
    if (!canLoadMerged) {
      if (!cfg.model_path) {
        showToast(
          "Set base model path (or merge once first) before Test in Chat",
          "error",
          6000,
        );
        return;
      }
      const mergeRes = await runMergeFromTrain(cfg);
      if (!mergeRes.ok) {
        trainLogLine(`Merge failed: ${mergeRes.error}`, "err");
        showToast(`Merge failed: ${mergeRes.error}`, "error", 6000);
        return;
      }
      canLoadMerged = true;
    }

    if (!canLoadMerged) {
      showToast("Merged model not found", "error");
      return;
    }

    trainLogLine(`load ${mergedPath}`, "", "cmd");
    // Stop any running LLM first
    await window.My.llm.stop();
    const res = await window.My.llm.start(mergedPath);
    if (!res.ok) {
      trainLogLine(`Failed to load: ${res.error}`, "err");
      showToast(`Load failed: ${res.error}`, "error", 6000);
      return;
    }
    trainLogLine(`Model ready: ${mergedPath}`, "ok");
    showToast("Trained model loaded — switching to Chat ✓", "success");
    activateScreen("chat");
  } finally {
    setTrainTestBusy(false);
    await maybeEnableTrainTestButton();
  }
});

window.My.train.onEvent((msg: BridgeMsg) => {
  switch (msg.type) {
    case "status":
      trainLogLine(`${msg["message"]}`, "", "info");
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
      trainLogLine(`epoch ${msg["epoch"]} / ${msg["total"]}`, "", "cmd");
      break;
    case "log":
      trainLogRaw(String(msg["text"] ?? ""));
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
          trainLogLine(t, "", "warn");
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
      void maybeEnableTrainTestButton();
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
  savedPromptsCache = await window.My.prompts.list();
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
  await window.My.prompts.save(entry);
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
  await window.My.prompts.delete(id);
  await refreshSavedPrompts();
  showToast("Prompt deleted", "info");
});

refreshSavedPrompts();

// ─────────────────────────────────────────────────────────────────────────────
// BOOKMIND (RAG) SCREEN
// ─────────────────────────────────────────────────────────────────────────────

interface BookCandidate {
  _id?: string;
  Title?: string;
  Subtitle?: string;
  Authors?: string[];
  Genres?: string[];
  Moods?: string[];
  Pacing?: string;
  Description?: string;
  AvgRating?: number;
  CoverImageUrl?: string;
  PageCount?: number;
  score?: number;
}

const booksState = {
  bridgeRunning: false,
  bridgeReady: false,
  querying: false,
};

const booksQueryInput = $<HTMLInputElement>("#books-query");
const booksUserIdInput = $<HTMLInputElement>("#books-user-id");
const booksGoodreadsInput = $<HTMLInputElement>("#books-goodreads");
const booksLimitInput = $<HTMLInputElement>("#books-limit");
const booksUseLlmInput = $<HTMLInputElement>("#books-use-llm");
const booksStatus = $<HTMLElement>("#books-status");
const booksStartBtn = $<HTMLButtonElement>("#books-start-btn");
const booksQueryBtn = $<HTMLButtonElement>("#books-query-btn");
const booksStopBtn = $<HTMLButtonElement>("#books-stop-btn");
const booksCandidatesList = $<HTMLElement>("#books-candidates-list");
const booksLlmOutput = $<HTMLElement>("#books-llm-output");

function updateBooksUI() {
  booksQueryBtn.disabled = !booksState.bridgeReady || booksState.querying;
  booksStopBtn.style.display = booksState.bridgeRunning ? "" : "none";
  booksStartBtn.textContent = booksState.bridgeReady
    ? "Reload Bridge"
    : booksState.bridgeRunning
      ? "Starting…"
      : "▶ Start Bridge";
  if (booksState.bridgeReady) {
    booksStatus.textContent = "BookMind bridge ready";
    booksStatus.classList.add("ok");
  } else if (booksState.bridgeRunning) {
    booksStatus.textContent = "Starting bridge…";
    booksStatus.classList.remove("ok");
  } else {
    booksStatus.textContent = "Bridge not started";
    booksStatus.classList.remove("ok");
  }
}

function renderBookCandidates(books: BookCandidate[]) {
  booksCandidatesList.innerHTML = "";
  if (!books.length) {
    booksCandidatesList.innerHTML =
      '<div style="color:var(--text3);font-size:12px;padding:20px 0">No matches found.</div>';
    return;
  }
  books.forEach((b) => {
    const card = document.createElement("div");
    card.className = "book-card";

    const cover = document.createElement("div");
    cover.className = "book-cover";
    if (b.CoverImageUrl) {
      const img = document.createElement("img");
      img.src = b.CoverImageUrl;
      img.alt = b.Title || "cover";
      img.onerror = () => {
        img.remove();
      };
      cover.appendChild(img);
    }

    const info = document.createElement("div");
    info.className = "book-info";

    const title = document.createElement("div");
    title.className = "bc-title";
    title.textContent = b.Title || "Untitled";
    title.title = b.Title || "";

    const author = document.createElement("div");
    author.className = "bc-author";
    const authors = b.Authors || [];
    author.textContent = authors.length ? authors.join(", ") : "Unknown author";

    const meta = document.createElement("div");
    meta.className = "bc-meta";
    const parts: string[] = [];
    const genres = (b.Genres || []).slice(0, 3);
    if (genres.length) parts.push(genres.join(" · "));
    if (typeof b.AvgRating === "number")
      parts.push(`★ ${b.AvgRating.toFixed(1)}`);
    if (b.Pacing) parts.push(b.Pacing);
    if (typeof b.score === "number")
      parts.push(
        `<span class="bc-score">${(b.score * 100).toFixed(0)}%</span>`,
      );
    meta.innerHTML = parts.join("&nbsp;&nbsp;•&nbsp;&nbsp;");

    info.appendChild(title);
    info.appendChild(author);
    info.appendChild(meta);

    // Expandable description
    const desc = document.createElement("div");
    desc.className = "bc-desc";
    const descText = (b.Description || "").trim();
    if (descText) {
      desc.textContent =
        descText.length > 500 ? descText.slice(0, 500) + "…" : descText;
      if (b.PageCount) {
        desc.textContent += ` (${b.PageCount} pages)`;
      }
    } else {
      desc.textContent = "No description available.";
    }
    info.appendChild(desc);

    // Click to toggle description
    card.addEventListener("click", () => {
      card.classList.toggle("expanded");
    });

    card.appendChild(cover);
    card.appendChild(info);
    booksCandidatesList.appendChild(card);
  });
}

booksStartBtn.addEventListener("click", async () => {
  if (booksState.bridgeRunning) {
    await window.My.books.stop();
  }
  booksState.bridgeRunning = true;
  booksState.bridgeReady = false;
  updateBooksUI();
  const res = await window.My.books.start();
  if (!res.ok) {
    booksState.bridgeRunning = false;
    updateBooksUI();
    showToast(`Failed to start BookMind: ${res.error}`, "error", 6000);
  }
});

booksStopBtn.addEventListener("click", async () => {
  await window.My.books.stop();
  booksState.bridgeRunning = false;
  booksState.bridgeReady = false;
  booksState.querying = false;
  updateBooksUI();
});

booksQueryBtn.addEventListener("click", async () => {
  if (!booksState.bridgeReady || booksState.querying) return;
  const query = booksQueryInput.value.trim();
  const userId = booksUserIdInput.value.trim();
  const goodreads = booksGoodreadsInput.value.trim();
  if (!query && !userId && !goodreads) {
    showToast("Enter a query, user id, or Goodreads handle", "info");
    return;
  }
  booksState.querying = true;
  updateBooksUI();

  booksCandidatesList.innerHTML =
    '<div style="color:var(--text3);font-size:12px;padding:20px 0">Searching…</div>';
  booksLlmOutput.innerHTML = "";

  const request: Record<string, unknown> = {
    query,
    limit: parseInt(booksLimitInput.value) || 10,
  };
  if (userId) request["user_id"] = userId;
  if (goodreads) request["goodreads_user"] = goodreads;
  if (booksUseLlmInput.checked && state.paths?.llmModel) {
    request["llm_model_dir"] = state.paths.llmModel;
  }

  const res = await window.My.books.query(request);
  if (!res.ok) {
    booksState.querying = false;
    updateBooksUI();
    showToast(`Query failed: ${res.error}`, "error", 5000);
  }
});

booksQueryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !booksQueryBtn.disabled) {
    booksQueryBtn.click();
  }
});

// Refresh the Books UI state whenever the user navigates to the screen
document
  .querySelector<HTMLElement>('.nav-btn[data-screen="books"]')
  ?.addEventListener("click", () => updateBooksUI());

// Initial paint of the Books UI state
updateBooksUI();

window.My.books.onEvent((msg: BridgeMsg) => {
  if (msg.type === "status") {
    const text = msg["message"] as string;
    booksStatus.textContent = text;
    if (/ready/i.test(text)) {
      booksState.bridgeReady = true;
      booksState.bridgeRunning = true;
      updateBooksUI();
    }
  } else if (msg.type === "goodreads_matched") {
    const fetched = Number(msg["fetched"] || 0);
    const matched = (msg["books"] as unknown[]) || [];
    showToast(
      `Goodreads: fetched ${fetched}, matched ${matched.length} to library`,
      "success",
      4500,
    );
  } else if (msg.type === "candidates") {
    const books = (msg["books"] || []) as BookCandidate[];
    renderBookCandidates(books);
  } else if (msg.type === "token") {
    const t = (msg["text"] || "") as string;
    booksLlmOutput.textContent = (booksLlmOutput.textContent || "") + t;
  } else if (msg.type === "done") {
    booksState.querying = false;
    updateBooksUI();
    if (!booksLlmOutput.textContent) {
      booksLlmOutput.innerHTML =
        '<span style="color:var(--text3);font-size:12px">Enable "Use local LLM" to get explanations.</span>';
    }
  } else if (msg.type === "error") {
    booksState.querying = false;
    updateBooksUI();
    const m = String(msg["message"] || "unknown error");
    showToast(`BookMind error: ${m}`, "error", 6000);
    booksCandidatesList.innerHTML = `<div style="color:var(--danger);font-size:12px;padding:10px 0">${m}</div>`;
  } else if (msg.type === "exit") {
    booksState.bridgeRunning = false;
    booksState.bridgeReady = false;
    booksState.querying = false;
    updateBooksUI();
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// SETTINGS SCREEN
// ─────────────────────────────────────────────────────────────────────────────

const cfgMongoUri = $<HTMLInputElement>("#cfg-mongo-uri");
const cfgMongoDb = $<HTMLInputElement>("#cfg-mongo-db");
const cfgEmbedModel = $<HTMLInputElement>("#cfg-embed-model");
const cfgVectorIndex = $<HTMLInputElement>("#cfg-vector-index");
const cfgGoodreadsUser = $<HTMLInputElement>("#cfg-goodreads-user");
const cfgTestBtn = $<HTMLButtonElement>("#cfg-test-mongo");
const cfgMongoStatus = $<HTMLElement>("#cfg-mongo-status");
const cfgMongoResult = $<HTMLElement>("#cfg-mongo-result");
const cfgSaveBtn = $<HTMLButtonElement>("#cfg-save-btn");
const cfgSaveStatus = $<HTMLElement>("#cfg-save-status");
const cfgToggleBtn = $<HTMLButtonElement>("#cfg-mongo-toggle");

// Load saved config into fields
async function loadSettings() {
  const cfg = await window.My.config.get();
  cfgMongoUri.value = cfg.mongoUri || "";
  cfgMongoDb.value = cfg.mongoDb || "bookmind";
  cfgEmbedModel.value =
    cfg.embedModel || "sentence-transformers/all-MiniLM-L6-v2";
  cfgVectorIndex.value = cfg.vectorIndex || "vs_books_embedding";
  cfgGoodreadsUser.value = cfg.goodreadsUser || "";
}

// Toggle password visibility
cfgToggleBtn.addEventListener("click", () => {
  if (cfgMongoUri.type === "password") {
    cfgMongoUri.type = "text";
    cfgToggleBtn.textContent = "Hide";
  } else {
    cfgMongoUri.type = "password";
    cfgToggleBtn.textContent = "Show";
  }
});

// Test connection
cfgTestBtn.addEventListener("click", async () => {
  const uri = cfgMongoUri.value.trim();
  const db = cfgMongoDb.value.trim() || "bookmind";
  if (!uri) {
    showToast("Enter a MongoDB URI first", "error");
    return;
  }
  cfgTestBtn.disabled = true;
  cfgMongoStatus.textContent = "Testing…";
  cfgMongoStatus.style.color = "var(--text3)";
  cfgMongoResult.style.display = "none";

  const res = await window.My.config.testMongo(uri, db);
  cfgTestBtn.disabled = false;

  if (res.ok) {
    cfgMongoStatus.textContent = "Connected ✓";
    cfgMongoStatus.style.color = "var(--success)";
    if (res.collections) {
      const lines = Object.entries(res.collections)
        .map(([name, count]) => `  ${name}: ${count} docs`)
        .join("\n");
      cfgMongoResult.textContent = `Database "${db}" collections:\n${lines}`;
      cfgMongoResult.style.display = "block";
    }
  } else {
    cfgMongoStatus.textContent = "Connection failed ✗";
    cfgMongoStatus.style.color = "var(--danger)";
    cfgMongoResult.textContent = res.error || "Unknown error";
    cfgMongoResult.style.display = "block";
  }
});

// Save
cfgSaveBtn.addEventListener("click", async () => {
  const patch = {
    mongoUri: cfgMongoUri.value.trim(),
    mongoDb: cfgMongoDb.value.trim() || "bookmind",
    embedModel:
      cfgEmbedModel.value.trim() || "sentence-transformers/all-MiniLM-L6-v2",
    vectorIndex: cfgVectorIndex.value.trim() || "vs_books_embedding",
    goodreadsUser: cfgGoodreadsUser.value.trim(),
  };
  await window.My.config.set(patch);
  cfgSaveStatus.textContent = "Saved ✓";
  showToast("Settings saved", "success");
  setTimeout(() => {
    cfgSaveStatus.textContent = "";
  }, 3000);
});

// Load settings on first visit to the screen
document
  .querySelector<HTMLElement>('.nav-btn[data-screen="settings"]')
  ?.addEventListener("click", () => loadSettings());

// Also load on startup to check if mongo is configured
loadSettings();

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
  const installed = await window.My.models.list();
  const installedNames = new Set(installed.map((m) => m.name.toLowerCase()));
  return ESSENTIALS.filter(
    (e) => !installedNames.has(e.targetDir.toLowerCase()),
  ).map((e) => e.id);
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
  if (allDone) {
    essentialsInstallAllBtn.textContent = "All set — let's go ✓";
    essentialsInstallAllBtn.disabled = false;
    // Clicking "All set" just closes the modal
    essentialsInstallAllBtn.onclick = () =>
      essentialsModal.classList.remove("open");
  } else {
    essentialsInstallAllBtn.textContent = "⬇ Install All";
    essentialsInstallAllBtn.disabled = false;
    essentialsInstallAllBtn.onclick = null; // falls through to installAllEssentials
  }
}

async function installEssential(id: string): Promise<void> {
  const state = essentialState[id];
  if (!state || state.installed || state.done || state.downloading) return;
  state.downloading = true;
  renderEssentials();

  return new Promise((resolve) => {
    const dispose = window.My.models.onDownloadEvent((msg: BridgeMsg) => {
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
    window.My.models.download(state.repoId, state.targetDir, state.type);
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
