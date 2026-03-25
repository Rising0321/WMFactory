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
const SHARED_CONTROL_PROFILES = {
  diamond: { invertX: true, invertY: true },
  matrixgame: { invertX: false, invertY: false },
  "open-oasis": { invertX: false, invertY: false },
  worldfm: { invertX: false, invertY: false },
  mineworld: { invertX: MINEWORLD_INVERT_CAMERA_X, invertY: MINEWORLD_INVERT_CAMERA_Y },
  "infinite-world": { invertX: false, invertY: false },
  vid2world: { invertX: VID2WORLD_INVERT_CAMERA_X, invertY: VID2WORLD_INVERT_CAMERA_Y },
  default: { invertX: INVERT_CAMERA_X, invertY: INVERT_CAMERA_Y },
};

function createControls() {
  return {
    w: false,
    a: false,
    s: false,
    d: false,
    camera_dx: 0,
    camera_dy: 0,
    l_click: false,
    r_click: false,
  };
}

function createSideRuntime() {
  return {
    controls: createControls(),
    mineworldCameraActive: false,
    mineworldCameraPointerId: null,
    mineworldCameraLastX: 0,
    mineworldCameraLastY: 0,
  };
}

const state = {
  models: [],
  leftModelId: null,
  rightModelId: null,
  leftLoaded: false,
  rightLoaded: false,
  seedImage: null,
  stepping: false,
  sessionActive: false,
  sides: {
    left: createSideRuntime(),
    right: createSideRuntime(),
  },
};

const el = {
  leftModelSelect: document.getElementById("leftModelSelect"),
  rightModelSelect: document.getElementById("rightModelSelect"),
  loadArenaBtn: document.getElementById("loadArenaBtn"),
  startBattleBtn: document.getElementById("startBattleBtn"),
  resetBtn: document.getElementById("resetBtn"),
  imageInput: document.getElementById("imageInput"),
  seedPreview: document.getElementById("seedPreview"),
  battleStatus: document.getElementById("battleStatus"),
  leftModelStatus: document.getElementById("leftModelStatus"),
  rightModelStatus: document.getElementById("rightModelStatus"),
  leftFrameView: document.getElementById("leftFrameView"),
  rightFrameView: document.getElementById("rightFrameView"),
  leftLabel: document.getElementById("leftLabel"),
  rightLabel: document.getElementById("rightLabel"),
  leftMeta: document.getElementById("leftMeta"),
  rightMeta: document.getElementById("rightMeta"),
  startOverlay: document.getElementById("startOverlay"),
  startFloatingBtn: document.getElementById("startFloatingBtn"),
  leftCameraStick: document.getElementById("leftCameraStick"),
  rightCameraStick: document.getElementById("rightCameraStick"),
  leftCameraKnob: document.getElementById("leftCameraKnob"),
  rightCameraKnob: document.getElementById("rightCameraKnob"),
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

function setBattleStatus(text, isError = false) {
  el.battleStatus.textContent = text;
  el.battleStatus.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function updateOverlay() {
  el.startOverlay.classList.toggle("hidden", state.sessionActive);
}

function getModelLabel(modelId) {
  return state.models.find((item) => item.id === modelId)?.label || modelId || "未选择";
}

function updateHeadings() {
  el.leftLabel.textContent = getModelLabel(state.leftModelId);
  el.rightLabel.textContent = getModelLabel(state.rightModelId);
}

function setSideLoadedMeta(side, payload) {
  const meta = `${payload.model_id} / GPU ${payload.visible_devices || payload.gpu_index}`;
  if (side === "left") {
    el.leftMeta.textContent = meta;
    el.leftModelStatus.textContent = `已加载到 ${payload.device}`;
  } else {
    el.rightMeta.textContent = meta;
    el.rightModelStatus.textContent = `已加载到 ${payload.device}`;
  }
}

function clearArenaFrames() {
  el.leftFrameView.src = state.seedImage || EMPTY_FRAME_DATA_URL;
  el.rightFrameView.src = state.seedImage || EMPTY_FRAME_DATA_URL;
}

function showSeedEverywhere() {
  if (!state.seedImage) {
    return;
  }
  el.seedPreview.src = state.seedImage;
  clearArenaFrames();
  state.sessionActive = false;
  updateOverlay();
}

function syncDistinctSelects(changedSide) {
  if (state.leftModelId !== state.rightModelId) {
    return;
  }
  const fallback = state.models.find((item) => item.id !== state.leftModelId)?.id;
  if (!fallback) {
    return;
  }
  if (changedSide === "left") {
    state.rightModelId = fallback;
    el.rightModelSelect.value = fallback;
  } else {
    state.leftModelId = fallback;
    el.leftModelSelect.value = fallback;
  }
}

function resetSideRuntime(side) {
  state.sides[side] = createSideRuntime();
}

function invalidateLoadedArena(reasonText) {
  state.leftLoaded = false;
  state.rightLoaded = false;
  state.sessionActive = false;
  resetSideRuntime("left");
  resetSideRuntime("right");
  el.leftModelStatus.textContent = "未加载";
  el.rightModelStatus.textContent = "未加载";
  el.leftMeta.textContent = "等待加载";
  el.rightMeta.textContent = "等待加载";
  updateAllControllerStyles();
  centerAllKnobs();
  updateOverlay();
  if (reasonText) {
    setBattleStatus(reasonText);
  }
}

async function loadModels() {
  const data = await api("/api/models");
  state.models = data.models || [];

  for (const select of [el.leftModelSelect, el.rightModelSelect]) {
    select.innerHTML = "";
    for (const model of state.models) {
      const opt = document.createElement("option");
      opt.value = model.id;
      opt.textContent = model.label;
      select.appendChild(opt);
    }
  }

  state.leftModelId = state.models[0]?.id || null;
  state.rightModelId = state.models[1]?.id || state.models[0]?.id || null;
  if (state.leftModelId) {
    el.leftModelSelect.value = state.leftModelId;
  }
  if (state.rightModelId) {
    el.rightModelSelect.value = state.rightModelId;
  }
  updateHeadings();
}

async function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function onLoadArena() {
  if (!state.leftModelId || !state.rightModelId) {
    setBattleStatus("模型列表为空，无法启动 WMArena。", true);
    return;
  }
  if (state.leftModelId === state.rightModelId) {
    setBattleStatus("第一阶段要求左右必须选择两个不同模型。", true);
    return;
  }

  setBattleStatus("正在为左右两侧分配独立 GPU 并加载模型...");
  el.leftModelStatus.textContent = "加载中...";
  el.rightModelStatus.textContent = "加载中...";
  try {
    const data = await api("/api/arena/load", {
      method: "POST",
      body: JSON.stringify({
        left_model_id: state.leftModelId,
        right_model_id: state.rightModelId,
      }),
    });
    state.leftLoaded = true;
    state.rightLoaded = true;
    setSideLoadedMeta("left", data.left);
    setSideLoadedMeta("right", data.right);
    setBattleStatus("双模型已就绪。上传相同初始图后即可开始对战。");
  } catch (err) {
    state.leftLoaded = false;
    state.rightLoaded = false;
    el.leftMeta.textContent = "加载失败";
    el.rightMeta.textContent = "加载失败";
    setBattleStatus(`加载失败: ${err.message}`, true);
  }
}

async function onStartBattle() {
  if (!state.seedImage) {
    setBattleStatus("请先上传一张共同的初始图像。", true);
    return;
  }
  if (!state.leftLoaded || !state.rightLoaded) {
    setBattleStatus("请先完成双模型加载。", true);
    return;
  }

  setBattleStatus("正在用同一张图初始化左右会话...");
  try {
    const data = await api("/api/arena/start", {
      method: "POST",
      body: JSON.stringify({ init_image_base64: state.seedImage }),
    });
    state.sessionActive = true;
    el.leftFrameView.src = `data:image/png;base64,${data.left.frame_base64}`;
    el.rightFrameView.src = `data:image/png;base64,${data.right.frame_base64}`;
    updateOverlay();
    setBattleStatus("对战已开始。现在左右两侧使用各自独立控制器。");
  } catch (err) {
    setBattleStatus(`启动失败: ${err.message}`, true);
  }
}

async function onResetBattle() {
  if (!state.sessionActive) {
    return;
  }
  try {
    const data = await api("/api/arena/reset", {
      method: "POST",
      body: JSON.stringify({ init_image_base64: state.seedImage }),
    });
    el.leftFrameView.src = `data:image/png;base64,${data.left.frame_base64}`;
    el.rightFrameView.src = `data:image/png;base64,${data.right.frame_base64}`;
    setBattleStatus("左右会话已同步重置。");
  } catch (err) {
    setBattleStatus(`重置失败: ${err.message}`, true);
  }
}

function isChunkedModel(modelId) {
  return ["yume", "infinite-world", "gamecraft", "worldplay", "lingbot-world"].includes(modelId);
}

function isLatencyModel(modelId) {
  return modelId === "vid2world";
}

function chunkedModelLabel(modelId) {
  const labels = {
    "infinite-world": "Infinite-World",
    yume: "YUME",
    gamecraft: "GameCraft",
    worldplay: "WorldPlay",
    "lingbot-world": "LingBot-World",
  };
  return labels[modelId] || "Chunked Model";
}

function modelIdForSide(side) {
  return side === "left" ? state.leftModelId : state.rightModelId;
}

function controlsForSide(side) {
  return state.sides[side].controls;
}

function neutralAction() {
  return {
    w: false,
    a: false,
    s: false,
    d: false,
    camera_dx: 0,
    camera_dy: 0,
    l_click: false,
    r_click: false,
  };
}

function effectiveActionForSide(side) {
  const modelId = modelIdForSide(side);
  const controls = { ...controlsForSide(side) };
  const cameraDeadzone = modelId === "mineworld" ? MINEWORLD_CAMERA_DEADZONE : CAMERA_DEADZONE;
  if (Math.abs(controls.camera_dx) <= cameraDeadzone) {
    controls.camera_dx = 0;
  }
  if (Math.abs(controls.camera_dy) <= cameraDeadzone) {
    controls.camera_dy = 0;
  }
  return controls;
}

function hasInputForSide(side) {
  const modelId = modelIdForSide(side);
  const cameraDeadzone = modelId === "mineworld" ? MINEWORLD_CAMERA_DEADZONE : CAMERA_DEADZONE;
  const controls = controlsForSide(side);
  return (
    !!controls.w ||
    !!controls.a ||
    !!controls.s ||
    !!controls.d ||
    !!controls.l_click ||
    !!controls.r_click ||
    Math.abs(controls.camera_dx) > cameraDeadzone ||
    Math.abs(controls.camera_dy) > cameraDeadzone
  );
}

async function stepLoop() {
  if (!state.sessionActive || state.stepping) {
    return;
  }

  const leftHasInput = hasInputForSide("left");
  const rightHasInput = hasInputForSide("right");
  if (!leftHasInput && !rightHasInput) {
    return;
  }

  const leftAction = leftHasInput ? effectiveActionForSide("left") : neutralAction();
  const rightAction = rightHasInput ? effectiveActionForSide("right") : neutralAction();
  state.stepping = true;

  try {
    if (leftHasInput && isChunkedModel(state.leftModelId)) {
      setBattleStatus(`${chunkedModelLabel(state.leftModelId)} 左侧正在生成下一段...`);
    } else if (rightHasInput && isChunkedModel(state.rightModelId)) {
      setBattleStatus(`${chunkedModelLabel(state.rightModelId)} 右侧正在生成下一段...`);
    }

    const data = await api("/api/arena/step", {
      method: "POST",
      body: JSON.stringify({
        left_action: leftAction,
        right_action: rightAction,
      }),
    });

    el.leftFrameView.src = `data:image/png;base64,${data.left.frame_base64}`;
    el.rightFrameView.src = `data:image/png;base64,${data.right.frame_base64}`;

    const leftLatency = Number(data?.left?.extra?.latency_ms || 0);
    const rightLatency = Number(data?.right?.extra?.latency_ms || 0);
    if (leftLatency > 0 || rightLatency > 0 || isLatencyModel(state.leftModelId) || isLatencyModel(state.rightModelId)) {
      const leftSeconds = leftLatency > 0 ? `${(leftLatency / 1000).toFixed(1)}s` : "-";
      const rightSeconds = rightLatency > 0 ? `${(rightLatency / 1000).toFixed(1)}s` : "-";
      setBattleStatus(`双侧 step 完成。左侧 ${leftSeconds}，右侧 ${rightSeconds}。`);
    }

    if (data.left.ended || data.left.truncated || data.right.ended || data.right.truncated) {
      setBattleStatus("至少一侧回合结束，正在同步重置。", true);
      await onResetBattle();
    }
  } catch (err) {
    setBattleStatus(`Step 失败: ${err.message}`, true);
  } finally {
    for (const side of ["left", "right"]) {
      if (modelIdForSide(side) === "mineworld") {
        controlsForSide(side).camera_dx = 0;
        controlsForSide(side).camera_dy = 0;
        paintMineWorldCamera(side, 0, 0);
      }
    }
    state.stepping = false;
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function cameraInversionForModel(modelId) {
  return SHARED_CONTROL_PROFILES[modelId] || SHARED_CONTROL_PROFILES.default;
}

function knobForSide(side) {
  return side === "left" ? el.leftCameraKnob : el.rightCameraKnob;
}

function stickForSide(side) {
  return side === "left" ? el.leftCameraStick : el.rightCameraStick;
}

function paintKnob(side, dx, dy) {
  const size = 160;
  const knobSize = 54;
  const center = size / 2;
  const maxRadius = (size - knobSize) / 2;
  const x = center - knobSize / 2 + dx * maxRadius;
  const y = center - knobSize / 2 + dy * maxRadius;
  const knob = knobForSide(side);
  knob.style.left = `${x}px`;
  knob.style.top = `${y}px`;
}

function paintMineWorldCamera(side, dx, dy) {
  paintKnob(side, dx, dy);
}

function centerAllKnobs() {
  paintKnob("left", 0, 0);
  paintKnob("right", 0, 0);
}

function updateAllControllerStyles() {
  document.querySelectorAll(".key").forEach((btn) => {
    const side = btn.dataset.side;
    const key = btn.dataset.key;
    btn.classList.toggle("active", !!controlsForSide(side)[key]);
  });
}

function bindWASDButtons() {
  document.querySelectorAll(".key").forEach((btn) => {
    const side = btn.dataset.side;
    const key = btn.dataset.key;
    const press = () => {
      controlsForSide(side)[key] = true;
      updateAllControllerStyles();
    };
    const release = () => {
      controlsForSide(side)[key] = false;
      updateAllControllerStyles();
    };
    btn.addEventListener("pointerdown", press);
    btn.addEventListener("pointerup", release);
    btn.addEventListener("pointercancel", release);
    btn.addEventListener("pointerleave", release);
  });
}

function bindCameraStick(side) {
  const stick = stickForSide(side);
  const size = 160;
  const knobSize = 54;
  const center = size / 2;
  const maxRadius = (size - knobSize) / 2;
  let active = false;

  const setFromPointer = (clientX, clientY) => {
    const rect = stick.getBoundingClientRect();
    const x = clientX - rect.left - center;
    const y = clientY - rect.top - center;
    const len = Math.hypot(x, y);
    const scale = len > maxRadius ? maxRadius / len : 1;
    const nx = (x * scale) / maxRadius;
    const ny = (y * scale) / maxRadius;
    const inv = cameraInversionForModel(modelIdForSide(side));
    controlsForSide(side).camera_dx = Number((inv.invertX ? -nx : nx).toFixed(3));
    controlsForSide(side).camera_dy = Number((inv.invertY ? -ny : ny).toFixed(3));
    paintKnob(side, nx, ny);
  };

  const resetStick = () => {
    controlsForSide(side).camera_dx = 0;
    controlsForSide(side).camera_dy = 0;
    paintKnob(side, 0, 0);
  };

  const resetMineWorldStick = () => {
    controlsForSide(side).camera_dx = 0;
    controlsForSide(side).camera_dy = 0;
    state.sides[side].mineworldCameraActive = false;
    state.sides[side].mineworldCameraPointerId = null;
    paintMineWorldCamera(side, 0, 0);
  };

  const updateMineWorldFromDelta = (deltaX, deltaY) => {
    const nx = clamp((deltaX / maxRadius) * MINEWORLD_CAMERA_DELTA_GAIN, -MINEWORLD_CAMERA_MAX_DELTA, MINEWORLD_CAMERA_MAX_DELTA);
    const ny = clamp((deltaY / maxRadius) * MINEWORLD_CAMERA_DELTA_GAIN, -MINEWORLD_CAMERA_MAX_DELTA, MINEWORLD_CAMERA_MAX_DELTA);
    const inv = cameraInversionForModel(modelIdForSide(side));
    controlsForSide(side).camera_dx = Number((inv.invertX ? -nx : nx).toFixed(3));
    controlsForSide(side).camera_dy = Number((inv.invertY ? -ny : ny).toFixed(3));
    paintMineWorldCamera(side, nx, ny);
  };

  stick.addEventListener("pointerdown", (e) => {
    if (modelIdForSide(side) === "mineworld") {
      state.sides[side].mineworldCameraActive = true;
      state.sides[side].mineworldCameraPointerId = e.pointerId;
      state.sides[side].mineworldCameraLastX = e.clientX;
      state.sides[side].mineworldCameraLastY = e.clientY;
      stick.setPointerCapture?.(e.pointerId);
      return;
    }
    active = true;
    setFromPointer(e.clientX, e.clientY);
  });

  window.addEventListener("pointermove", (e) => {
    if (modelIdForSide(side) === "mineworld") {
      if (!state.sides[side].mineworldCameraActive || state.sides[side].mineworldCameraPointerId !== e.pointerId) {
        return;
      }
      const deltaX = e.clientX - state.sides[side].mineworldCameraLastX;
      const deltaY = e.clientY - state.sides[side].mineworldCameraLastY;
      state.sides[side].mineworldCameraLastX = e.clientX;
      state.sides[side].mineworldCameraLastY = e.clientY;
      updateMineWorldFromDelta(deltaX, deltaY);
      return;
    }
    if (!active) {
      return;
    }
    setFromPointer(e.clientX, e.clientY);
  });

  const end = (e) => {
    if (modelIdForSide(side) === "mineworld") {
      if (!state.sides[side].mineworldCameraActive) {
        return;
      }
      if (e && state.sides[side].mineworldCameraPointerId !== null && e.pointerId !== state.sides[side].mineworldCameraPointerId) {
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
  el.leftModelSelect.addEventListener("change", () => {
    state.leftModelId = el.leftModelSelect.value;
    syncDistinctSelects("left");
    updateHeadings();
    invalidateLoadedArena("模型选择已变化，请重新加载双模型。");
  });

  el.rightModelSelect.addEventListener("change", () => {
    state.rightModelId = el.rightModelSelect.value;
    syncDistinctSelects("right");
    updateHeadings();
    invalidateLoadedArena("模型选择已变化，请重新加载双模型。");
  });

  el.imageInput.addEventListener("change", async (e) => {
    const [file] = e.target.files || [];
    if (!file) {
      return;
    }
    state.seedImage = await readFileAsDataUrl(file);
    showSeedEverywhere();
    setBattleStatus(`已设置共同初始图像: ${file.name}`);
  });

  el.loadArenaBtn.addEventListener("click", onLoadArena);
  el.startBattleBtn.addEventListener("click", onStartBattle);
  el.startFloatingBtn.addEventListener("click", onStartBattle);
  el.resetBtn.addEventListener("click", onResetBattle);

  bindWASDButtons();
  bindCameraStick("left");
  bindCameraStick("right");
}

async function boot() {
  try {
    await loadModels();
    bindEvents();
    el.seedPreview.src = EMPTY_FRAME_DATA_URL;
    el.leftFrameView.src = EMPTY_FRAME_DATA_URL;
    el.rightFrameView.src = EMPTY_FRAME_DATA_URL;
    centerAllKnobs();
    updateOverlay();
    setInterval(stepLoop, 80);
    setBattleStatus("就绪：左右选择两个不同模型，上传共同初始图，然后分别使用下方控制器操作。");
  } catch (err) {
    setBattleStatus(`初始化失败: ${err.message}`, true);
  }
}

boot();
