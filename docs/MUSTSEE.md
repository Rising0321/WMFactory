# MUSTSEE: WMFactory 前后端统一接口规范（新增 World Model 必读）

## 1. 目标与范围

本项目的目标是统一不同 World Model 的体验入口：

- 同一前端页面可切换不同模型
- 支持输入图像（上传）或从数据集随机抽图
- 支持 WASD + 相机控制（拖动/摇杆）
- 后端通过“适配器”接入不同模型，不依赖模型作者自带 UI 入口

本文件定义当前已经实现并需要保持兼容的接口合同。

## 2. 推荐总架构（必须遵守）

- 前端: `frontend/web/`（纯 HTML/CSS/JS）
- 网关后端: `frontend/server.py`（FastAPI，统一 API，对前端唯一入口）
- 模型服务层: 每个模型一个独立服务（独立 Python 环境/容器）
- 模型适配层:
  - 网关侧 adapter（调用模型服务，不直接 import 模型代码）
  - 模型服务侧 runtime（在对应环境中真正推理）

核心原则：

1. 前端只调用统一 API，不直接调用具体模型代码。
2. 每个模型运行在自己的环境中，禁止在网关进程混装多模型依赖。
3. 新模型接入时，优先新增“模型服务 + 网关 adapter”，不改前端交互协议。

## 3. 当前实现状态

DIAMOND 已完成服务化改造：

- 模型服务: `services/diamond/app.py`
- 网关适配器: `frontend/adapters/diamond_adapter.py`（HTTP 调服务）

后续新增模型请直接复制这个模式，不要回退到“网关进程内直连模型”。

## 4. 网关 Adapter 统一接口（硬性要求）

所有新模型 Adapter 必须实现 `WorldModelAdapter`：

```python
class WorldModelAdapter(ABC):
    model_id: str
    def load(self) -> Dict[str, Any]: ...
    def start_session(self, init_image_bytes: Optional[bytes]) -> Dict[str, Any]: ...
    def reset_session(self, init_image_bytes: Optional[bytes]) -> Dict[str, Any]: ...
    def step(self, action: Dict[str, Any]) -> StepResult: ...
```

`StepResult` 固定字段：

- `frame_base64: str` (PNG，不带 data URL 前缀)
- `reward: float`
- `ended: bool`
- `truncated: bool`
- `extra: Dict[str, Any]`

补充要求（服务化场景）：

1. `load/start/reset/step` 内部应通过 HTTP 调模型服务，不直接跑模型。
2. adapter 负责把网关统一协议映射成模型服务协议（可 1:1 透传）。
3. adapter 必须实现超时、重试、错误翻译（统一成网关 `HTTPException`）。

### 4.1 服务生命周期与启动时机（硬性要求）

1. 模型服务默认采用懒启动（lazy start）：
- 网关进程启动时不强制启动所有模型服务。
- 用户点击“确认加载模型”（`POST /api/models/load`）时，adapter 才检查服务健康并按需启动服务。

2. 允许手动预启动模型服务，但 adapter 必须兼容“服务已运行”场景。

3. 推荐环境变量：
- `WM_<MODEL>_AUTOSTART=1|0`：是否允许网关自动拉起模型服务。
- `WM_<MODEL>_STARTUP_TIMEOUT`：服务启动等待时间。

## 5. 网关 API 合同（前端依赖）

### 5.1 模型相关

#### `GET /api/models`
返回模型列表。最少字段：

- `id` (唯一，和 adapter key 一致)
- `label`
- `status`
- `description`

#### `POST /api/models/load`
请求：

```json
{ "model_id": "diamond" }
```

返回至少包含：

- `model_id`
- `status` (`loaded`/`already_loaded`)
- `device`

### 5.2 数据集相关

#### `GET /api/datasets`
返回数据集列表，每项：

- `id`
- `label`
- `num_images`

#### `POST /api/datasets/random-image`
请求：

```json
{ "dataset_id": "CSGO" }
```

返回：

```json
{
  "dataset_id": "CSGO",
  "file": "data/xxx.png or ...",
  "image_base64": "data:image/png;base64,..."
}
```

注意：

- 普通数据集默认从 `data/<dataset>/` 搜索图片扩展名（png/jpg/jpeg/webp/bmp）。
- DIAMOND 的 `CSGO` 支持特殊 fallback：从 spawn 的 `full_res.npy` 随机抽一帧。

### 5.3 会话相关

#### `POST /api/sessions/start`
请求：

```json
{
  "model_id": "diamond",
  "init_image_base64": "data:image/png;base64,..." 
}
```

`init_image_base64` 可空（但前端当前会先要求用户选图）。

返回：

```json
{
  "session_id": "uuid",
  "frame_base64": "..."
}
```

#### `POST /api/sessions/step`
请求：

```json
{
  "session_id": "...",
  "action": {
    "w": true,
    "a": false,
    "s": false,
    "d": true,
    "camera_dx": 0.15,
    "camera_dy": -0.1,
    "l_click": false,
    "r_click": false,
    "space": false,
    "ctrl": false,
    "shift": false,
    "weapon": 0,
    "reload": false
  }
}
```

返回：

```json
{
  "session_id": "...",
  "frame_base64": "...",
  "reward": 0.0,
  "ended": false,
  "truncated": false,
  "extra": {}
}
```

#### `POST /api/sessions/reset`
请求：

```json
{
  "session_id": "...",
  "init_image_base64": "data:image/png;base64,..."
}
```

返回同 `start`。

## 6. 模型服务 API 合同（网关调用）

每个模型服务必须暴露以下 HTTP 接口（建议 `POST` + JSON）：

1. `POST /health`
- 返回 `{ "ok": true, "model_id": "...", "ready": true/false }`

2. `POST /load`
- 入参可含模型权重路径、设备偏好、运行参数
- 返回至少：`model_id/status/device`

3. `POST /sessions/start`
- 入参：`init_image_base64`（可空）及可选初始化参数
- 返回：`session_id/frame_base64`

4. `POST /sessions/step`
- 入参：`session_id/action`
- 返回：`session_id/frame_base64/reward/ended/truncated/extra`

5. `POST /sessions/reset`
- 入参：`session_id/init_image_base64`（可空）
- 返回：`session_id/frame_base64`

6. `POST /datasets/random-image`（可选）
- 返回同网关随机图字段

要求：

1. 字段名和类型尽量与网关统一协议一致，减少映射复杂度。
2. 所有错误返回统一 `{ \"detail\": \"...\" }`。
3. 接口必须幂等或可重试（尤其 `load`）。

## 7. 状态管理约束（当前版本）

当前后端是“单活跃模型 + 单活跃会话”实现：

- `state.active_model_id` 表示当前活跃模型
- `session_id` 必须匹配活跃 adapter 的 runtime session
- 不支持多用户并发隔离（后续可扩展）

新增模型时可保持该约束，除非你同时重构会话管理。

## 8. 动作协议规范

前端发送统一动作 JSON，adapter 负责映射到各模型内部动作空间。

DIAMOND 映射规则（参考实现）：

- 键位: `w/a/s/d/space/ctrl/shift/weapon/reload`
- 点击: `l_click/r_click`
- 视角: `camera_dx/camera_dy`，输入范围约 `[-1, 1]`
- adapter 内部可离散化（DIAMOND 映射到鼠标桶 one-hot）

要求：

1. 新模型必须兼容上述通用 action keys（可忽略不需要的字段）。
2. 如模型需要额外动作，允许在 `action` 中扩展键，但不能破坏既有键含义。

## 9. 图像与张量规范

统一输入图像流程：

- 前端传 `data:image/...;base64,...`
- 后端 `_decode_image` 解码 bytes
- Adapter 自行 resize/normalize 到模型需要的分辨率

统一输出帧流程：

- adapter 输出 PNG 的 base64 字符串（无前缀）给 `frame_base64`
- 前端用 `data:image/png;base64,${frame_base64}` 显示

## 10. 新模型接入步骤（Checklist）

1. 为新模型创建独立运行环境（conda/venv/docker 任一）。
2. 在该环境内实现模型服务（`/health /load /sessions/*`）。
3. 在 `frontend/adapters/` 新建网关 adapter，通过 HTTP 调模型服务。
4. 在 `frontend/adapters/__init__.py` 导出该 adapter。
5. 在 `frontend/server.py` 的 `AppState.adapters` 注册：`"<model_id>": <Adapter>()`。
6. 在 `GET /api/models` 中加入该模型元信息。
7. 如模型有特定数据集随机图格式，在网关或模型服务增加 fallback。
8. 保证 `start/reset/step` 返回字段与本文件一致。
9. 本地验证：
   - `GET /api/models` 能看到新模型
   - `POST /api/models/load` 成功
   - `POST /api/sessions/start` 返回 session
   - `POST /api/sessions/step` 可连续返回帧

## 11. 错误处理规范

- 不支持模型: `404` + `detail: Unsupported model 'xxx'`
- 无活跃会话: `400` + `detail: No active model session`
- 会话不匹配: `400` + `detail: Unknown or expired session_id`
- 运行时异常: `500` + `detail: <真实错误>`

要求：错误必须保留 `detail` 文本，便于前端直接展示。

## 12. 可观测性与日志回传（硬性要求）

1. 网关必须打印每次模型服务调用的关键日志，格式统一：
- `[{model_id}] request -> <path>`
- `[{model_id}] request <- <path> ok`
- `[{model_id}] request <- <path> failed status=<code> detail=<detail>`

2. 如果模型服务由网关自动拉起，服务 stdout/stderr 需要被转发到网关输出，并添加模型前缀：
- `[{model_id}] <service log line>`

3. 高频接口（如 `/sessions/step`）必须做日志节流，避免刷屏。
- 推荐环境变量：`WM_<MODEL>_STEP_LOG_EVERY`。

## 13. 环境与部署约定

仓库结构（建议演进）：

- 模型代码: `models/`
- 数据集: `data/`
- 网关: `frontend/`
- 模型服务: `services/<model_id>/`（建议新增）

DIAMOND 资产路径优先级（示例）：

1. 环境变量 `DIAMOND_CKPT_PATH` + `DIAMOND_SPAWN_DIR`
2. 本地 `models/diamond/csgo/...`
3. HuggingFace 下载 fallback

服务发现建议：

1. 用环境变量声明模型服务地址，如 `WM_DIAMOND_URL=http://127.0.0.1:9001`。
2. 网关 adapter 只读 URL，不关心具体环境管理细节。

## 14. 你提到的方案 vs 更稳方案

你提的“模型作为服务 + POST 传参数和结果”是正确方向。  
更稳的落地方式是：

1. `Gateway + Model Worker` 双层架构（而不是前端直连模型服务）
2. 模型服务独立部署（每个模型一个环境/容器）
3. 网关统一鉴权、限流、会话和协议转换

原因：前端不会感知后端异构，后续可替换任何模型实现。

## 15. 未来扩展建议（不影响当前合同）

- 多会话/多用户并发管理（session store）
- WebSocket 流式帧推送
- 模型能力描述 schema（支持的动作键、最大 FPS、分辨率）
- 统一 benchmark/eval hook（为 LLMArena 类比较准备）

---

如果你在新对话里继续开发，请先阅读：

- `MUSTSEE.md`（本文件）
- `frontend/server.py`
- `frontend/adapters/base.py`

优先遵守本文件中的接口合同，避免破坏现有前端。
