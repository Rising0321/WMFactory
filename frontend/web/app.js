const state = {
  modelId: "diamond",
  modelLoaded: false,
  sessionId: null,
  seedImage: null,
  currentSource: "upload",
  controls: {
    w: false,
    a: false,
    s: false,
    d: false,
    camera_dx: 0,
    camera_dy: 0,
    l_click: false,
    r_click: false,
  },
  cameraHeld: { up: false, down: false, left: false, right: false },
  stepping: false,
};

const EMPTY_FRAME_DATA_URL =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
const CAMERA_DEADZONE = 0.08;
const MINEWORLD_CAMERA_DEADZONE = 0.015;

const el = {
  modelSelect: document.getElementById("modelSelect"),
  loadModelBtn: document.getElementById("loadModelBtn"),
  modelStatus: document.getElementById("modelStatus"),
  sourceSelect: document.getElementById("sourceSelect"),
  randomBtn: document.getElementById("randomBtn"),
  imageInput: document.getElementById("imageInput"),
  resetBtn: document.getElementById("resetBtn"),
  startFloatingBtn: document.getElementById("startFloatingBtn"),
  startOverlay: document.getElementById("startOverlay"),
  frameView: document.getElementById("frameView"),
  gameStatus: document.getElementById("gameStatus"),
  camUp: document.getElementById("camUp"),
  camDown: document.getElementById("camDown"),
  camLeft: document.getElementById("camLeft"),
  camRight: document.getElementById("camRight"),
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function setModelStatus(text, isError = false) {
  el.modelStatus.textContent = text;
  el.modelStatus.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function setGameStatus(text, isError = false) {
  el.gameStatus.textContent = text;
  el.gameStatus.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function updateStartOverlay() {
  const hide = !!state.sessionId;
  el.startOverlay.classList.toggle("hidden", hide);
}

function showSeedInViewport() {
  if (!state.seedImage) {
    return;
  }
  state.sessionId = null;
  el.frameView.src = state.seedImage;
  updateStartOverlay();
}

async function loadModelsAndDatasets() {
  const [modelData, datasetData] = await Promise.all([api("/api/models"), api("/api/datasets")]);

  el.modelSelect.innerHTML = "";
  for (const m of modelData.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    el.modelSelect.appendChild(opt);
  }
  state.modelId = el.modelSelect.value;

  el.sourceSelect.innerHTML = "";
  const upload = document.createElement("option");
  upload.value = "upload";
  upload.textContent = "输入图像";
  el.sourceSelect.appendChild(upload);

  for (const ds of datasetData.datasets) {
    const opt = document.createElement("option");
    opt.value = ds.id;
    opt.textContent = `${ds.label} (${ds.num_images})`;
    el.sourceSelect.appendChild(opt);
  }
  syncSourceForModel(state.modelId);
}

async function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(file);
  });
}

function resolveInitImage() {
  if (state.currentSource === "upload") {
    return state.seedImage;
  }
  return state.seedImage;
}

function preferredDatasetForModel(modelId) {
  if (modelId === "mineworld") {
    return "minecraft";
  }
  if (modelId === "diamond") {
    return "CSGO";
  }
  if (modelId === "vid2world") {
    return "CSGO";
  }
  return null;
}

function isChunkedModel(modelId) {
  return (
    modelId === "yume" ||
    modelId === "infinite-world" ||
    modelId === "worldplay" ||
    modelId === "lingbot-world-fast" ||
    modelId === "matrixgame3"
  );
}

function isLatencyModel(modelId) {
  return modelId === "vid2world";
}

function chunkedModelLabel(modelId) {
  if (modelId === "infinite-world") {
    return "Infinite-World";
  }
  if (modelId === "yume") {
    return "YUME";
  }
  if (modelId === "worldplay") {
    return "HY-WorldPlay 5B";
  }
  if (modelId === "lingbot-world-fast") {
    return "LingBot-World-Fast";
  }
  if (modelId === "matrixgame3") {
    return "Matrix-Game 3.0";
  }
  return "Chunked Model";
}

function sourceOptionExists(value) {
  return Array.from(el.sourceSelect.options).some((opt) => opt.value === value);
}

function syncSourceForModel(modelId) {
  const preferred = preferredDatasetForModel(modelId);
  if (preferred && sourceOptionExists(preferred)) {
    el.sourceSelect.value = preferred;
  } else {
    el.sourceSelect.value = "upload";
  }
  state.currentSource = el.sourceSelect.value;
  el.imageInput.disabled = state.currentSource !== "upload";
}

async function onLoadModel() {
  state.modelId = el.modelSelect.value;
  setModelStatus("正在加载模型到 GPU...");
  try {
    const result = await api("/api/models/load", {
      method: "POST",
      body: JSON.stringify({ model_id: state.modelId }),
    });
    state.modelLoaded = true;
    setModelStatus(`模型已加载: ${result.model_id} @ ${result.device}`);
  } catch (err) {
    setModelStatus(`加载失败: ${err.message}`, true);
  }
}

async function onRandomImage() {
  const datasetId = el.sourceSelect.value;
  if (datasetId === "upload") {
    setGameStatus("当前模式是输入图像，请先上传图片。", true);
    return;
  }
  try {
    const data = await api("/api/datasets/random-image", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId }),
    });
    state.seedImage = data.image_base64;
    showSeedInViewport();
    setGameStatus(`已从 ${datasetId} 随机抽图: ${data.file}`);
  } catch (err) {
    setGameStatus(`随机图失败: ${err.message}`, true);
  }
}

async function onStartSession() {
  if (!state.seedImage) {
    setGameStatus("请先上传图像/选择数据集。", true);
    return;
  }
  if (!state.modelLoaded) {
    setGameStatus("请先加载模型。", true);
    return;
  }
  setGameStatus("会话启动中...");

  try {
    const data = await api("/api/sessions/start", {
      method: "POST",
      body: JSON.stringify({
        model_id: state.modelId,
        init_image_base64: resolveInitImage(),
      }),
    });
    state.sessionId = data.session_id;
    el.frameView.src = `data:image/png;base64,${data.frame_base64}`;
    updateStartOverlay();
    setGameStatus(`会话已启动: ${state.sessionId.slice(0, 8)}...`);
  } catch (err) {
    setGameStatus(`启动失败: ${err.message}`, true);
  }
}

async function onResetSession() {
  if (!state.sessionId) {
    return;
  }
  try {
    const data = await api("/api/sessions/reset", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        init_image_base64: resolveInitImage(),
      }),
    });
    state.sessionId = data.session_id;
    el.frameView.src = `data:image/png;base64,${data.frame_base64}`;
    updateStartOverlay();
    setGameStatus("会话已重置");
  } catch (err) {
    setGameStatus(`重置失败: ${err.message}`, true);
  }
}

async function stepLoop() {
  if (!state.sessionId || state.stepping) {
    return;
  }
  const cameraDeadzone = state.modelId === "mineworld" ? MINEWORLD_CAMERA_DEADZONE : CAMERA_DEADZONE;
  if (Math.abs(state.controls.camera_dx) <= cameraDeadzone) {
    state.controls.camera_dx = 0;
  }
  if (Math.abs(state.controls.camera_dy) <= cameraDeadzone) {
    state.controls.camera_dy = 0;
  }
  const hasInput =
    !!state.controls.w ||
    !!state.controls.a ||
    !!state.controls.s ||
    !!state.controls.d ||
    !!state.controls.l_click ||
    !!state.controls.r_click ||
    Math.abs(state.controls.camera_dx) > cameraDeadzone ||
    Math.abs(state.controls.camera_dy) > cameraDeadzone;

  if (!hasInput) {
    return;
  }

  const action = { ...state.controls };
  state.stepping = true;
  try {
    if (isChunkedModel(state.modelId)) {
      setGameStatus(`${chunkedModelLabel(state.modelId)} 正在生成下一段视频，请等待当前 chunk 完成...`);
    }
    const data = await api("/api/sessions/step", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        action,
      }),
    });
    el.frameView.src = `data:image/png;base64,${data.frame_base64}`;
    if (isChunkedModel(state.modelId)) {
      const latencyMs = Number(data?.extra?.latency_ms || 0);
      const seconds = latencyMs > 0 ? (latencyMs / 1000).toFixed(1) : null;
      const motion = [data?.extra?.movement_key, data?.extra?.camera_key, data?.extra?.move, data?.extra?.view].filter(Boolean).join(" / ");
      const modelLabel = chunkedModelLabel(state.modelId);
      setGameStatus(
        seconds
          ? `${modelLabel} 已生成下一段: ${motion || "动作已应用"}，耗时 ${seconds}s`
          : `${modelLabel} 已生成下一段视频`
      );
    } else if (isLatencyModel(state.modelId)) {
      const latencyMs = Number(data?.extra?.latency_ms || 0);
      if (latencyMs > 0) {
        setGameStatus(`Vid2World 已生成下一帧，耗时 ${(latencyMs / 1000).toFixed(1)}s`);
      }
    }
    if (data.ended || data.truncated) {
      setGameStatus("回合结束，自动重置。", true);
      await onResetSession();
    }
  } catch (err) {
    setGameStatus(`Step失败: ${err.message}`, true);
  } finally {
    state.stepping = false;
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function updateWASDStyles() {
  document.querySelectorAll(".key").forEach((btn) => {
    const key = btn.dataset.key;
    btn.classList.toggle("active", !!state.controls[key]);
  });
}

function updateCameraFromHeld() {
  const dx = (state.cameraHeld.right ? 1 : 0) + (state.cameraHeld.left ? -1 : 0);
  const dy = (state.cameraHeld.down ? 1 : 0) + (state.cameraHeld.up ? -1 : 0);
  state.controls.camera_dx = clamp(dx, -1, 1);
  state.controls.camera_dy = clamp(dy, -1, 1);
  document.querySelectorAll(".cam-btn").forEach((btn) => {
    const dir = btn.dataset.dir;
    btn.classList.toggle("active", !!state.cameraHeld[dir]);
  });
}

function bindKeyboard() {
  const downMap = {
    w: "w",
    a: "a",
    s: "s",
    d: "d",
  };

  window.addEventListener("keydown", (e) => {
    const k = downMap[e.key.toLowerCase()];
    if (!k) {
      return;
    }
    state.controls[k] = true;
    updateWASDStyles();
  });

  window.addEventListener("keyup", (e) => {
    const k = downMap[e.key.toLowerCase()];
    if (!k) {
      return;
    }
    state.controls[k] = false;
    updateWASDStyles();
  });

  window.addEventListener("blur", () => {
    state.controls.w = false;
    state.controls.a = false;
    state.controls.s = false;
    state.controls.d = false;
    state.controls.l_click = false;
    state.controls.r_click = false;
    state.controls.camera_dx = 0;
    state.controls.camera_dy = 0;
    state.cameraHeld.up = false;
    state.cameraHeld.down = false;
    state.cameraHeld.left = false;
    state.cameraHeld.right = false;
    updateCameraFromHeld();
    updateWASDStyles();
  });
}

function bindWASDButtons() {
  document.querySelectorAll(".key").forEach((btn) => {
    const key = btn.dataset.key;

    const press = () => {
      state.controls[key] = true;
      updateWASDStyles();
    };
    const release = () => {
      state.controls[key] = false;
      updateWASDStyles();
    };

    btn.addEventListener("pointerdown", press);
    btn.addEventListener("pointerup", release);
    btn.addEventListener("pointercancel", release);
    btn.addEventListener("pointerleave", release);
  });
}

function bindCameraButtons() {
  const map = [
    { el: el.camUp, dir: "up" },
    { el: el.camDown, dir: "down" },
    { el: el.camLeft, dir: "left" },
    { el: el.camRight, dir: "right" },
  ];
  for (const { el: btn, dir } of map) {
    if (!btn) continue;
    btn.dataset.dir = dir;
    const press = () => {
      state.cameraHeld[dir] = true;
      updateCameraFromHeld();
    };
    const release = () => {
      state.cameraHeld[dir] = false;
      updateCameraFromHeld();
    };
    btn.addEventListener("pointerdown", press);
    btn.addEventListener("pointerup", release);
    btn.addEventListener("pointercancel", release);
    btn.addEventListener("pointerleave", release);
  }
  updateCameraFromHeld();
}

function bindEvents() {
  el.modelSelect.addEventListener("change", () => {
    state.modelId = el.modelSelect.value;
    syncSourceForModel(state.modelId);
  });

  el.sourceSelect.addEventListener("change", () => {
    state.currentSource = el.sourceSelect.value;
    const isUpload = state.currentSource === "upload";
    el.imageInput.disabled = !isUpload;
  });

  el.imageInput.addEventListener("change", async (e) => {
    const [file] = e.target.files || [];
    if (!file) {
      return;
    }
    state.seedImage = await readFileAsDataUrl(file);
    showSeedInViewport();
    setGameStatus(`已选择输入图像: ${file.name}`);
  });

  el.loadModelBtn.addEventListener("click", onLoadModel);
  el.randomBtn.addEventListener("click", onRandomImage);
  el.startFloatingBtn.addEventListener("click", onStartSession);
  el.resetBtn.addEventListener("click", onResetSession);

  bindKeyboard();
  bindWASDButtons();
  bindCameraButtons();
}

async function boot() {
  try {
    await loadModelsAndDatasets();
    bindEvents();
    el.frameView.src = EMPTY_FRAME_DATA_URL;
    setInterval(stepLoop, 80);
    setGameStatus("就绪：先加载模型，再选择输入图启动会话。");
    updateStartOverlay();
  } catch (err) {
    setGameStatus(`初始化失败: ${err.message}`, true);
  }
}

boot();
