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
const DEFAULT_LANG = "en";
const CLIENT_ID_KEY = "wmarena_client_id";

const I18N = {
  en: {
    html_lang: "en",
    toggle_lang: "中文",
    anonymous_mode: "Anonymous Mode",
    anonymous_mode_on: "Anonymous On",
    hero_title: "World Model Arena",
    hero_desc: "Choose two models, upload one shared seed image, and drive each side with its own controller.",
    load_left: "Load Left Side",
    load_right: "Load Right Side",
    reset_session: "Reset Session",
    left_model: "Left Model",
    right_model: "Right Model",
    seed_image: "Seed Image",
    battle_view: "Battle View",
    battle_note: "Each side sends its own action JSON, and the next input waits until both returns finish.",
    left_controller: "Left Controller",
    right_controller: "Right Controller",
    camera: "Camera",
    overlay_desc: "Load both models and upload a shared seed image before starting.",
    enter_world: "Enter World",
    dual_controllers: "Dual Controllers",
    dual_controllers_desc: "Each model gets its own controller so mismatched action scales do not drift the scenes apart.",
    vote_title: "Which side do you support?",
    vote_left: "Support Left",
    vote_right: "Support Right",
    vote_both: "Support Both",
    vote_neither: "Support Neither",
    not_loaded: "Not loaded",
    waiting_load: "Waiting to load",
    load_failed: "Load failed",
    loading_side: "Loading...",
    side_none: "No model selected",
    anonymous_left: "Anonymous Model A",
    anonymous_right: "Anonymous Model B",
    side_meta: "{model} / GPU {gpu}",
    side_loaded: "Loaded on {device}",
    choose_models_then_seed: "Choose two models first, then upload a seed image.",
    boot_ready: "Ready: choose two different models, upload one shared seed image, then control each side separately.",
    status_model_list_empty: "Model list is empty. WMArena cannot start.",
    status_models_must_differ: "Phase 1 requires two different models on the left and right.",
    status_loading_left: "Assigning the left GPU group and loading the left model...",
    status_loading_right: "Assigning the right GPU group and loading the right model...",
    status_side_loaded: "{side} is ready. Load the other side before entering the world.",
    status_models_ready: "Both models are ready. Upload the same seed image to start the arena.",
    status_load_failed: "Load failed: {message}",
    status_seed_required: "Upload one shared seed image first.",
    status_load_required: "Load both models first.",
    status_starting: "Initializing both sessions from the same seed image...",
    status_started: "Battle started. Left and right now use separate controllers.",
    status_start_failed: "Start failed: {message}",
    status_reset_done: "Both sessions have been reset in sync.",
    status_reset_failed: "Reset failed: {message}",
    status_model_changed: "Model selection changed. Reload the affected side.",
    status_seed_selected: "Shared seed image selected: {name}",
    status_chunk_left: "{model} is generating the next left-side chunk...",
    status_chunk_right: "{model} is generating the next right-side chunk...",
    status_step_done: "Both steps finished. Left {left}, right {right}.",
    status_round_end: "At least one side ended. Resetting both sessions.",
    status_step_failed: "Step failed: {message}",
    status_anonymous_enabled: "Anonymous mode enabled. Load will randomize and hide both models.",
    status_anonymous_disabled: "Anonymous mode disabled. Load will use the selected models.",
    status_vote_saved: "Vote saved. Resetting the current round.",
    status_vote_failed: "Vote failed: {message}",
    status_vote_reveal: "Vote saved. Anonymous identities are now revealed.",
  },
  zh: {
    html_lang: "zh-CN",
    toggle_lang: "EN",
    anonymous_mode: "匿名模式",
    anonymous_mode_on: "匿名中",
    hero_title: "世界模型竞技场",
    hero_desc: "选择两个模型，上传同一张初始图，并分别用各自控制器驱动左右两侧。",
    load_left: "加载左侧",
    load_right: "加载右侧",
    reset_session: "重置会话",
    left_model: "左侧模型",
    right_model: "右侧模型",
    seed_image: "初始图像",
    battle_view: "对战视图",
    battle_note: "左右两侧分别发送各自 action JSON，并在两侧都返回后才接受下一次输入。",
    left_controller: "左侧控制器",
    right_controller: "右侧控制器",
    camera: "视角",
    overlay_desc: "先加载双模型并上传共同初始图像，再开始对战。",
    enter_world: "进入世界",
    dual_controllers: "双控制器",
    dual_controllers_desc: "每个模型各用一套控制器，避免动作尺度不同导致画面逐步漂移。",
    vote_title: "你支持哪一边？",
    vote_left: "支持左边",
    vote_right: "支持右边",
    vote_both: "都支持",
    vote_neither: "都不支持",
    not_loaded: "未加载",
    waiting_load: "等待加载",
    load_failed: "加载失败",
    loading_side: "加载中...",
    side_none: "未选择模型",
    anonymous_left: "匿名模型 A",
    anonymous_right: "匿名模型 B",
    side_meta: "{model} / GPU {gpu}",
    side_loaded: "已加载到 {device}",
    choose_models_then_seed: "先选择两个模型，再上传初始图像。",
    boot_ready: "就绪：左右选择两个不同模型，上传共同初始图，然后分别使用下方控制器操作。",
    status_model_list_empty: "模型列表为空，无法启动 WMArena。",
    status_models_must_differ: "第一阶段要求左右必须选择两个不同模型。",
    status_loading_left: "正在为左侧分配独立 GPU 并加载模型...",
    status_loading_right: "正在为右侧分配独立 GPU 并加载模型...",
    status_side_loaded: "{side} 已就绪，请继续加载另一侧后再进入世界。",
    status_models_ready: "双模型已就绪。上传相同初始图后即可开始对战。",
    status_load_failed: "加载失败: {message}",
    status_seed_required: "请先上传一张共同的初始图像。",
    status_load_required: "请先完成双模型加载。",
    status_starting: "正在用同一张图初始化左右会话...",
    status_started: "对战已开始。现在左右两侧使用各自独立控制器。",
    status_start_failed: "启动失败: {message}",
    status_reset_done: "左右会话已同步重置。",
    status_reset_failed: "重置失败: {message}",
    status_model_changed: "模型选择已变化，请重新加载对应侧。",
    status_seed_selected: "已设置共同初始图像: {name}",
    status_chunk_left: "{model} 左侧正在生成下一段...",
    status_chunk_right: "{model} 右侧正在生成下一段...",
    status_step_done: "双侧 step 完成。左侧 {left}，右侧 {right}。",
    status_round_end: "至少一侧回合结束，正在同步重置。",
    status_step_failed: "Step 失败: {message}",
    status_anonymous_enabled: "匿名模式已开启。加载时会随机并隐藏左右模型。",
    status_anonymous_disabled: "匿名模式已关闭。加载时会使用当前选择的模型。",
    status_vote_saved: "投票已记录，正在重置当前回合。",
    status_vote_failed: "投票失败: {message}",
    status_vote_reveal: "投票已记录，匿名模型真实身份已揭晓。",
  },
};

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
    device: "",
    gpu: "",
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
  lang: DEFAULT_LANG,
  clientId: null,
  anonymousMode: false,
  anonymousRevealed: false,
  roundId: null,
  voteVisible: false,
  voteSubmitted: false,
  hasRoundInteraction: false,
  lastBattleStatus: { key: "choose_models_then_seed", vars: {}, isError: false },
  sides: {
    left: createSideRuntime(),
    right: createSideRuntime(),
  },
};

const el = {
  leftModelSelect: document.getElementById("leftModelSelect"),
  rightModelSelect: document.getElementById("rightModelSelect"),
  loadLeftBtn: document.getElementById("loadLeftBtn"),
  loadRightBtn: document.getElementById("loadRightBtn"),
  resetBtn: document.getElementById("resetBtn"),
  langToggleBtn: document.getElementById("langToggleBtn"),
  anonymousModeBtn: document.getElementById("anonymousModeBtn"),
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
  voteCard: document.getElementById("voteCard"),
};

function t(key, vars = {}) {
  const template = I18N[state.lang]?.[key] || I18N[DEFAULT_LANG][key] || key;
  return template.replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? ""));
}

function ensureClientId() {
  let clientId = window.localStorage.getItem(CLIENT_ID_KEY);
  if (!clientId) {
    clientId = `client-${crypto.randomUUID()}`;
    window.localStorage.setItem(CLIENT_ID_KEY, clientId);
  }
  state.clientId = clientId;
}

function applyStaticTranslations() {
  document.documentElement.lang = t("html_lang");
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  el.langToggleBtn.textContent = t("toggle_lang");
  el.anonymousModeBtn.textContent = state.anonymousMode ? t("anonymous_mode_on") : t("anonymous_mode");
}

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

function setBattleStatusKey(key, vars = {}, isError = false) {
  state.lastBattleStatus = { key, vars, isError };
  setBattleStatus(t(key, vars), isError);
}

function setVoteVisible(visible) {
  state.voteVisible = visible;
  el.voteCard.classList.toggle("hidden", !visible);
}

function resetVoteState() {
  state.voteVisible = false;
  state.voteSubmitted = false;
  state.hasRoundInteraction = false;
  setVoteVisible(false);
}

function updateOverlay() {
  el.startOverlay.classList.toggle("hidden", state.sessionActive);
}

function displayLabelForSide(side) {
  if (state.anonymousMode && !state.anonymousRevealed && (state.leftLoaded || state.rightLoaded)) {
    return side === "left" ? t("anonymous_left") : t("anonymous_right");
  }
  const modelId = side === "left" ? state.leftModelId : state.rightModelId;
  return state.models.find((item) => item.id === modelId)?.label || modelId || t("side_none");
}

function renderModelSelectors() {
  const sideConfigs = [
    { side: "left", select: el.leftModelSelect, modelId: state.leftModelId, anonymousKey: "anonymous_left" },
    { side: "right", select: el.rightModelSelect, modelId: state.rightModelId, anonymousKey: "anonymous_right" },
  ];

  for (const item of sideConfigs) {
    item.select.innerHTML = "";
    if (state.anonymousMode && !state.anonymousRevealed) {
      const opt = document.createElement("option");
      opt.value = item.modelId || item.side;
      opt.textContent = t(item.anonymousKey);
      item.select.appendChild(opt);
      item.select.value = opt.value;
      continue;
    }
    for (const model of state.models) {
      const opt = document.createElement("option");
      opt.value = model.id;
      opt.textContent = model.label;
      item.select.appendChild(opt);
    }
    if (item.modelId) {
      item.select.value = item.modelId;
    }
  }
}

function renderSideUi(side) {
  const runtime = state.sides[side];
  const labelEl = side === "left" ? el.leftLabel : el.rightLabel;
  const metaEl = side === "left" ? el.leftMeta : el.rightMeta;
  const statusEl = side === "left" ? el.leftModelStatus : el.rightModelStatus;
  const loaded = side === "left" ? state.leftLoaded : state.rightLoaded;
  const modelId = side === "left" ? state.leftModelId : state.rightModelId;

  labelEl.textContent = displayLabelForSide(side);
  const metaModel = state.anonymousMode && !state.anonymousRevealed ? displayLabelForSide(side) : modelId;
  metaEl.textContent = loaded ? t("side_meta", { model: metaModel, gpu: runtime.gpu }) : t("waiting_load");
  statusEl.textContent = loaded ? t("side_loaded", { device: runtime.device }) : t("not_loaded");
}

function renderLanguage() {
  applyStaticTranslations();
  renderModelSelectors();
  renderSideUi("left");
  renderSideUi("right");
  setBattleStatus(t(state.lastBattleStatus.key, state.lastBattleStatus.vars), state.lastBattleStatus.isError);
  setVoteVisible(state.voteVisible);
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
    return null;
  }
  const fallback = state.models.find((item) => item.id !== state.leftModelId)?.id;
  if (!fallback) {
    return null;
  }
  if (changedSide === "left") {
    state.rightModelId = fallback;
    el.rightModelSelect.value = fallback;
    return "right";
  } else {
    state.leftModelId = fallback;
    el.leftModelSelect.value = fallback;
    return "left";
  }
}

function resetSideRuntime(side) {
  state.sides[side] = createSideRuntime();
}

function sideTitleKey(side) {
  return side === "left" ? "left_model" : "right_model";
}

function invalidateLoadedArena(statusKey) {
  state.leftLoaded = false;
  state.rightLoaded = false;
  state.sessionActive = false;
  state.anonymousRevealed = false;
  state.roundId = null;
  resetSideRuntime("left");
  resetSideRuntime("right");
  resetVoteState();
  updateAllControllerStyles();
  centerAllKnobs();
  clearArenaFrames();
  updateOverlay();
  renderSideUi("left");
  renderSideUi("right");
  if (statusKey) {
    setBattleStatusKey(statusKey);
  }
}

function invalidateSide(side, statusKey) {
  if (side === "left") {
    state.leftLoaded = false;
  } else {
    state.rightLoaded = false;
  }
  state.sessionActive = false;
  state.anonymousRevealed = false;
  state.roundId = null;
  resetSideRuntime(side);
  resetVoteState();
  updateAllControllerStyles();
  centerAllKnobs();
  clearArenaFrames();
  updateOverlay();
  renderSideUi("left");
  renderSideUi("right");
  if (statusKey) {
    setBattleStatusKey(statusKey);
  }
}

async function loadModels() {
  const data = await api("/api/models");
  state.models = data.models || [];

  for (const select of [el.leftModelSelect, el.rightModelSelect]) {
    select.innerHTML = "";
  }

  state.leftModelId = state.models[0]?.id || null;
  state.rightModelId = state.models[1]?.id || state.models[0]?.id || null;
  if (state.leftModelId) {
    el.leftModelSelect.value = state.leftModelId;
  }
  if (state.rightModelId) {
    el.rightModelSelect.value = state.rightModelId;
  }
  renderModelSelectors();
  renderSideUi("left");
  renderSideUi("right");
}

function updateAnonymousUi() {
  el.anonymousModeBtn.classList.toggle("active", state.anonymousMode);
  el.leftModelSelect.disabled = state.anonymousMode;
  el.rightModelSelect.disabled = state.anonymousMode;
  renderLanguage();
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
  const side = "left";
  return onLoadSide(side);
}

async function onLoadSide(side) {
  if (!state.anonymousMode) {
    const selectedModelId = side === "left" ? state.leftModelId : state.rightModelId;
    const otherModelId = side === "left" ? state.rightModelId : state.leftModelId;
    if (!selectedModelId) {
      setBattleStatusKey("status_model_list_empty", {}, true);
      return;
    }
    if (otherModelId && selectedModelId === otherModelId) {
      setBattleStatusKey("status_models_must_differ", {}, true);
      return;
    }
  }

  setBattleStatusKey(side === "left" ? "status_loading_left" : "status_loading_right");
  if (side === "left") {
    el.leftModelStatus.textContent = t("loading_side");
  } else {
    el.rightModelStatus.textContent = t("loading_side");
  }
  try {
    const data = await api("/api/arena/load-side", {
      method: "POST",
      body: JSON.stringify({
        side,
        model_id: side === "left" ? state.leftModelId : state.rightModelId,
        anonymous_mode: state.anonymousMode,
        client_id: state.clientId,
      }),
    });
    state.leftLoaded = !!data.left.loaded;
    state.rightLoaded = !!data.right.loaded;
    state.leftModelId = data.left.model_id;
    state.rightModelId = data.right.model_id;
    if (data.left.model_id) {
      el.leftModelSelect.value = data.left.model_id;
    }
    if (data.right.model_id) {
      el.rightModelSelect.value = data.right.model_id;
    }
    state.sides.left.device = data.left.device || "";
    state.sides.right.device = data.right.device || "";
    state.sides.left.gpu = data.left.visible_devices || String(data.left.gpu_index ?? "");
    state.sides.right.gpu = data.right.visible_devices || String(data.right.gpu_index ?? "");
    state.anonymousMode = !!data.anonymous_mode;
    state.anonymousRevealed = !!data.anonymous_revealed;
    resetVoteState();
    updateAnonymousUi();
    renderSideUi("left");
    renderSideUi("right");
    if (state.leftLoaded && state.rightLoaded) {
      setBattleStatusKey("status_models_ready");
    } else {
      setBattleStatusKey("status_side_loaded", { side: t(sideTitleKey(side)) });
    }
  } catch (err) {
    if (side === "left") {
      state.leftLoaded = false;
      el.leftMeta.textContent = t("load_failed");
    } else {
      state.rightLoaded = false;
      el.rightMeta.textContent = t("load_failed");
    }
    setBattleStatusKey("status_load_failed", { message: err.message }, true);
  }
}

async function onStartBattle() {
  if (!state.seedImage) {
    setBattleStatusKey("status_seed_required", {}, true);
    return;
  }
  if (!state.leftLoaded || !state.rightLoaded) {
    setBattleStatusKey("status_load_required", {}, true);
    return;
  }

  setBattleStatusKey("status_starting");
  try {
    const data = await api("/api/arena/start", {
      method: "POST",
      body: JSON.stringify({ init_image_base64: state.seedImage }),
    });
    state.sessionActive = true;
    state.roundId = data.round_id || null;
    state.anonymousMode = !!data.anonymous_mode;
    state.anonymousRevealed = !!data.anonymous_revealed;
    resetVoteState();
    el.leftFrameView.src = `data:image/png;base64,${data.left.frame_base64}`;
    el.rightFrameView.src = `data:image/png;base64,${data.right.frame_base64}`;
    updateOverlay();
    renderSideUi("left");
    renderSideUi("right");
    setBattleStatusKey("status_started");
  } catch (err) {
    setBattleStatusKey("status_start_failed", { message: err.message }, true);
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
    state.roundId = data.round_id || null;
    state.anonymousMode = !!data.anonymous_mode;
    state.anonymousRevealed = !!data.anonymous_revealed;
    resetVoteState();
    el.leftFrameView.src = `data:image/png;base64,${data.left.frame_base64}`;
    el.rightFrameView.src = `data:image/png;base64,${data.right.frame_base64}`;
    renderSideUi("left");
    renderSideUi("right");
    setBattleStatusKey("status_reset_done");
  } catch (err) {
    setBattleStatusKey("status_reset_failed", { message: err.message }, true);
  }
}

async function submitVote(voteOption) {
  if (state.voteSubmitted || !state.sessionActive) {
    return;
  }
  state.voteSubmitted = true;
  try {
    const data = await api("/api/arena/vote", {
      method: "POST",
      body: JSON.stringify({
        vote_option: voteOption,
        client_id: state.clientId,
      }),
    });
    state.anonymousMode = !!data.anonymous_mode;
    state.anonymousRevealed = !!data.anonymous_revealed;
    state.leftModelId = data.left.model_id;
    state.rightModelId = data.right.model_id;
    renderSideUi("left");
    renderSideUi("right");
    await onResetBattle();
    setBattleStatusKey(state.anonymousMode ? "status_vote_reveal" : "status_vote_saved");
  } catch (err) {
    state.voteSubmitted = false;
    setBattleStatusKey("status_vote_failed", { message: err.message }, true);
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
      setBattleStatusKey("status_chunk_left", { model: chunkedModelLabel(state.leftModelId) });
    } else if (rightHasInput && isChunkedModel(state.rightModelId)) {
      setBattleStatusKey("status_chunk_right", { model: chunkedModelLabel(state.rightModelId) });
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

    if (!state.hasRoundInteraction) {
      state.hasRoundInteraction = true;
      setVoteVisible(true);
    }

    const leftLatency = Number(data?.left?.extra?.latency_ms || 0);
    const rightLatency = Number(data?.right?.extra?.latency_ms || 0);
    if (leftLatency > 0 || rightLatency > 0 || isLatencyModel(state.leftModelId) || isLatencyModel(state.rightModelId)) {
      const leftSeconds = leftLatency > 0 ? `${(leftLatency / 1000).toFixed(1)}s` : "-";
      const rightSeconds = rightLatency > 0 ? `${(rightLatency / 1000).toFixed(1)}s` : "-";
      setBattleStatusKey("status_step_done", { left: leftSeconds, right: rightSeconds });
    }

    if (data.left.ended || data.left.truncated || data.right.ended || data.right.truncated) {
      setBattleStatusKey("status_round_end", {}, true);
      await onResetBattle();
    }
  } catch (err) {
    setBattleStatusKey("status_step_failed", { message: err.message }, true);
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
    const adjustedSide = syncDistinctSelects("left");
    renderSideUi("left");
    renderSideUi("right");
    if (adjustedSide) {
      invalidateLoadedArena("status_model_changed");
      return;
    }
    invalidateSide("left", "status_model_changed");
  });

  el.rightModelSelect.addEventListener("change", () => {
    state.rightModelId = el.rightModelSelect.value;
    const adjustedSide = syncDistinctSelects("right");
    renderSideUi("left");
    renderSideUi("right");
    if (adjustedSide) {
      invalidateLoadedArena("status_model_changed");
      return;
    }
    invalidateSide("right", "status_model_changed");
  });

  el.imageInput.addEventListener("change", async (e) => {
    const [file] = e.target.files || [];
    if (!file) {
      return;
    }
    state.seedImage = await readFileAsDataUrl(file);
    showSeedEverywhere();
    setBattleStatusKey("status_seed_selected", { name: file.name });
  });

  el.langToggleBtn.addEventListener("click", () => {
    state.lang = state.lang === "en" ? "zh" : "en";
    renderLanguage();
  });

  el.anonymousModeBtn.addEventListener("click", () => {
    state.anonymousMode = !state.anonymousMode;
    state.anonymousRevealed = false;
    updateAnonymousUi();
    invalidateLoadedArena(state.anonymousMode ? "status_anonymous_enabled" : "status_anonymous_disabled");
  });

  el.loadLeftBtn.addEventListener("click", () => onLoadSide("left"));
  el.loadRightBtn.addEventListener("click", () => onLoadSide("right"));
  el.startFloatingBtn.addEventListener("click", onStartBattle);
  el.resetBtn.addEventListener("click", onResetBattle);

  document.querySelectorAll(".vote-btn").forEach((btn) => {
    btn.addEventListener("click", () => submitVote(btn.dataset.vote));
  });

  bindWASDButtons();
  bindCameraStick("left");
  bindCameraStick("right");
}

async function boot() {
  try {
    ensureClientId();
    applyStaticTranslations();
    await loadModels();
    bindEvents();
    updateAnonymousUi();
    el.seedPreview.src = EMPTY_FRAME_DATA_URL;
    el.leftFrameView.src = EMPTY_FRAME_DATA_URL;
    el.rightFrameView.src = EMPTY_FRAME_DATA_URL;
    centerAllKnobs();
    updateOverlay();
    setInterval(stepLoop, 80);
    setBattleStatusKey("choose_models_then_seed");
  } catch (err) {
    setBattleStatus(String(err.message || err), true);
  }
}

boot();
