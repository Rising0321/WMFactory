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
  mineworldCameraActive: false,
  mineworldCameraPointerId: null,
  mineworldCameraLastX: 0,
  mineworldCameraLastY: 0,
  stepping: false,
};

const EMPTY_FRAME_DATA_URL =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
const CAMERA_DEADZONE = 0.08;
const INVERT_CAMERA_X = true;
const INVERT_CAMERA_Y = true;
const MINEWORLD_INVERT_CAMERA_X = false;
const MINEWORLD_INVERT_CAMERA_Y = false;
const VID2WORLD_INVERT_CAMERA_X = false;
const VID2WORLD_INVERT_CAMERA_Y = false;
const MINEWORLD_CAMERA_DEADZONE = 0.015;
const MINEWORLD_CAMERA_DELTA_GAIN = 0.3;
const MINEWORLD_CAMERA_MAX_DELTA = 0.35;

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
  cameraStick: document.getElementById("cameraStick"),
  cameraKnob: document.getElementById("cameraKnob"),
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
  return modelId === "yume" || modelId === "infinite-world" || modelId === "gamecraft" || modelId === "worldplay" || modelId === "lingbot-world" || modelId === "matrixgame3";
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
  if (modelId === "gamecraft") {
    return "GameCraft";
  }
  if (modelId === "worldplay") {
    return "WorldPlay";
  }
  if (modelId === "lingbot-world") {
    return "LingBot-World";
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
    if (state.modelId === "mineworld") {
      state.controls.camera_dx = 0;
      state.controls.camera_dy = 0;
      paintMineWorldCamera(0, 0);
    }
    state.stepping = false;
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function cameraInversionForModel() {
  if (state.modelId === "diamond") {
    return { invertX: false, invertY: false };
  }
  if (state.modelId === "matrixgame") {
    return { invertX: false, invertY: false };
  }
  if (state.modelId === "matrixgame3") {
    return { invertX: false, invertY: true };
  }
  if (state.modelId === "open-oasis") {
    return { invertX: false, invertY: false };
  }
  if (state.modelId === "worldfm") {
    return { invertX: false, invertY: false };
  }
  if (state.modelId === "mineworld") {
    return { invertX: MINEWORLD_INVERT_CAMERA_X, invertY: MINEWORLD_INVERT_CAMERA_Y };
  }
  if (state.modelId === "infinite-world") {
    return { invertX: false, invertY: false };
  }
  if (state.modelId === "vid2world") {
    return { invertX: VID2WORLD_INVERT_CAMERA_X, invertY: VID2WORLD_INVERT_CAMERA_Y };
  }
  return { invertX: INVERT_CAMERA_X, invertY: INVERT_CAMERA_Y };
}

function paintMineWorldCamera(dx, dy) {
  const size = 160;
  const knobSize = 54;
  const center = size / 2;
  const maxRadius = (size - knobSize) / 2;
  const kx = center - knobSize / 2 + dx * maxRadius;
  const ky = center - knobSize / 2 + dy * maxRadius;
  el.cameraKnob.style.left = `${kx}px`;
  el.cameraKnob.style.top = `${ky}px`;
}

function updateWASDStyles() {
  document.querySelectorAll(".key").forEach((btn) => {
    const key = btn.dataset.key;
    btn.classList.toggle("active", !!state.controls[key]);
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
    state.mineworldCameraActive = false;
    state.mineworldCameraPointerId = null;
    paintMineWorldCamera(0, 0);
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

function bindCameraStick() {
  const stick = el.cameraStick;
  const knob = el.cameraKnob;
  const size = 160;
  const knobSize = 54;
  const center = size / 2;
  const maxRadius = (size - knobSize) / 2;

  let active = false;

  const paint = (dx, dy) => {
    const kx = center - knobSize / 2 + dx * maxRadius;
    const ky = center - knobSize / 2 + dy * maxRadius;
    knob.style.left = `${kx}px`;
    knob.style.top = `${ky}px`;
  };

  const setFromPointer = (clientX, clientY) => {
    const rect = stick.getBoundingClientRect();
    const x = clientX - rect.left - center;
    const y = clientY - rect.top - center;
    const len = Math.hypot(x, y);
    const scale = len > maxRadius ? maxRadius / len : 1;

    const nx = (x * scale) / maxRadius;
    const ny = (y * scale) / maxRadius;

    const inv = cameraInversionForModel();
    const mappedX = inv.invertX ? -nx : nx;
    const mappedY = inv.invertY ? -ny : ny;
    state.controls.camera_dx = Number(mappedX.toFixed(3));
    state.controls.camera_dy = Number(mappedY.toFixed(3));
    paint(nx, ny);
  };

  const resetStick = () => {
    state.controls.camera_dx = 0;
    state.controls.camera_dy = 0;
    paint(0, 0);
  };

  const resetMineWorldStick = () => {
    state.controls.camera_dx = 0;
    state.controls.camera_dy = 0;
    state.mineworldCameraActive = false;
    state.mineworldCameraPointerId = null;
    paintMineWorldCamera(0, 0);
  };

  const updateMineWorldFromDelta = (deltaX, deltaY) => {
    const nx = clamp((deltaX / maxRadius) * MINEWORLD_CAMERA_DELTA_GAIN, -MINEWORLD_CAMERA_MAX_DELTA, MINEWORLD_CAMERA_MAX_DELTA);
    const ny = clamp((deltaY / maxRadius) * MINEWORLD_CAMERA_DELTA_GAIN, -MINEWORLD_CAMERA_MAX_DELTA, MINEWORLD_CAMERA_MAX_DELTA);
    const inv = cameraInversionForModel();
    const mappedX = inv.invertX ? -nx : nx;
    const mappedY = inv.invertY ? -ny : ny;
    state.controls.camera_dx = Number(mappedX.toFixed(3));
    state.controls.camera_dy = Number(mappedY.toFixed(3));
    paintMineWorldCamera(nx, ny);
  };

  stick.addEventListener("pointerdown", (e) => {
    if (state.modelId === "mineworld") {
      state.mineworldCameraActive = true;
      state.mineworldCameraPointerId = e.pointerId;
      state.mineworldCameraLastX = e.clientX;
      state.mineworldCameraLastY = e.clientY;
      stick.setPointerCapture?.(e.pointerId);
      return;
    }
    active = true;
    setFromPointer(e.clientX, e.clientY);
  });

  window.addEventListener("pointermove", (e) => {
    if (state.modelId === "mineworld") {
      if (!state.mineworldCameraActive || state.mineworldCameraPointerId !== e.pointerId) {
        return;
      }
      const deltaX = e.clientX - state.mineworldCameraLastX;
      const deltaY = e.clientY - state.mineworldCameraLastY;
      state.mineworldCameraLastX = e.clientX;
      state.mineworldCameraLastY = e.clientY;
      updateMineWorldFromDelta(deltaX, deltaY);
      return;
    }
    if (!active) {
      return;
    }
    setFromPointer(e.clientX, e.clientY);
  });

  const end = (e) => {
    if (state.modelId === "mineworld") {
      if (!state.mineworldCameraActive) {
        return;
      }
      if (e && state.mineworldCameraPointerId !== null && e.pointerId !== state.mineworldCameraPointerId) {
        return;
      }
      resetMineWorldStick();
      return;
    }
    if (!active) {
      return;
    }
    active = false;
    resetStick();
  };

  window.addEventListener("pointerup", end);
  window.addEventListener("pointercancel", end);

  resetStick();
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
  bindCameraStick();
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
