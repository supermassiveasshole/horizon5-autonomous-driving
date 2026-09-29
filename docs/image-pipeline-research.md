# 实时图像管线调研与改造建议

调研日期：2026-09-30。目标是让模型持续取得足够新、时间含义明确的数值图像输入；允许采集、预处理和存档丢帧，缺少合格历史时跳过该次决策。本文是调研与设计建议，不表示已经实现或通过实机验收。未采集新画面、运行参考模型、下载权重或控制车辆。

证据分为三类：**代码事实**来自锁定源码；**已有本机证据**来自既有记录的只读重算；**设计建议**是下一版本应实现、仍需验证的契约。参考项目锁定为 [VisionAI `be74b8def76e8c3247f91e310c5375db1e494756`](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/tree/be74b8def76e8c3247f91e310c5375db1e494756)，本仓库审计基线为 `a889f7117e0bb02f76d634b7eb9410de4fb88dfe`。分支页和 raw URL 的缓存内容曾有差异，以下 VisionAI 行号以固定 SHA 实际源码为准。

## 结论与已确认要求

已按用户在 2026-09-30 的决定，选定 **DXGI Desktop Duplication 采集 → 最新帧槽 → 独立预处理 → 带源时间的短历史环 → 常驻推理 worker → 独立动作监督器**。模型直接接收 RGB 数组或 tensor；JPEG/NVENC 仅作为异步存档、观看分支。Python 首版拟用 DXcam，显式指定 `backend="dxgi"`；WGC 不再作为首选或实现前置。完成该结构后，再依据本机测量决定是否实现 D3D11→CUDA 的全 GPU 预处理。这里确认的是实施选择，当前运行代码尚未切换后端。

用户已明确要求，以下不是可选优化：

- capture、resize 和模型取图分别调度，模型不等待文件落盘或读取 JPEG。
- 主采集后端使用 DXGI Desktop Duplication，选定目标显示器及其所属 adapter；不依赖包装库的隐式后端选择。
- **内存中的 JPEG 也不作为模型接口。** 在线主路径不得包含图像编码、图像解码或文件读写。
- 只取最近、满足时间要求的几帧；保留每帧真实时间信息。不能靠按帧编号、重复旧图、等待补齐旧任务伪造固定帧率。
- 偶发丢帧、短暂缺图或录像写盘拥塞不能自动终止整次尝试；不够本次观测就跳过本次决策。
- 实时性由图像年龄、推理截止时间、动作有效期和有界内存共同保障，不能用“绝不丢帧”代替。

固定时间间隔选帧和向模型提供时间信息可以同时使用。前者限制输入分布，后者表达不可避免的残余抖动；二者均不能替代新鲜度门槛和动作过期处理。

## VisionAI 实际做了什么

以下是代码事实，不是 README 的性能承诺。

| 环节 | 固定源码中的实现 | 对本项目的意义 |
| --- | --- | --- |
| 采集 | `windows-capture` 的 `WindowsCapture(window_name=...)`，`start_free_threaded()`，对应 WGC 窗口捕获路径 | 采集回调与主推理循环分开；本项目选择 DXGI，无需复制其后端 |
| 预处理 | 回调内 `cv2.resize(..., INTER_AREA)` | resize 仍占用采集回调，尚未三阶段解耦 |
| 缓存 | `deque(maxlen=240)`，保存缩小的 BGR 数组和时间 | 有限内存，满后淘汰最旧帧，不等待消费者 |
| 时间 | resize 完成后调用 `perf_counter()` | 是应用处理后的时间，不是原始呈现时间 |
| 选帧 | 以最新缓存时间为锚，对目标时刻取最近邻 | 不顺序追赶积压画面 |
| 输入 | BGR→RGB，CPU float32/255、拼接、转 tensor、`.to(device)` | 数值输入，无 JPEG 往返 |
| 推理 | 主循环同步推理，结果 `.cpu().numpy()` | 单次推理完成再继续，并非异步 GPU 流水线 |
| 节拍 | 未用完周期则 sleep | 默认 30 Hz 是目标，非实测保证 |

来源：[缓存](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L252-L258)、[采集与推理](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L518-L584)、[循环节拍](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L666-L668)。有界 deque 的覆盖行为也见 [Python 官方文档](https://docs.python.org/3/library/collections.html#collections.deque)。

配置为 320×180、三帧、间隔 100 ms，覆盖约 200 ms；模型状态字段没有各帧 age/dt。源码只要求缓存非空，没有最近邻误差上限、不同源帧校验、最新帧最大年龄或“这次输入已用过”判断。因此刚启动只有一帧时可以重复填三槽；采集停滞但未关闭时，可以继续推理旧图。遥测也没有独立接收年龄检查。其值得借鉴的是内存历史与最新时间选帧，不能直接继承其异常策略。[输入配置](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/data/models/meta.json#L1-L19)、[选帧与状态输入](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L543-L584)、[遥测接收](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L261-L287)。

其在线驾驶源码没有图像文件编码、落盘、再读盘链路；但没有公开录制器，不能把其训练数据录制性能当作已知。[驾驶代码](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py)、[数据与录制器说明](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/README.md#-training)。

### WGC 并不自动等于 GPU tensor 零拷贝

VisionAI 的 `windows-capture` 依赖没有锁版本，无法据此确定作者当时的二进制行为。[requirements.txt](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/requirements.txt#L15-L20)

后端判断来自其具体调用：Python `WindowsCapture` 创建 `NativeWindowsCapture`，后者通过 `GraphicsCaptureApiHandler` 启动 WGC。当前库另有独立的 DXGI 接口，但 VisionAI 没有调用它；不能因为新版依赖支持 DXGI，就将该项目描述为 DXGI 采集。[Python 包装](https://github.com/NiiightmareXD/windows-capture/blob/c7d106448eb9d9b251345c39047711e1cd408ae2/windows-capture-python/windows_capture/__init__.py#L140-L250)、[原生启动](https://github.com/NiiightmareXD/windows-capture/blob/c7d106448eb9d9b251345c39047711e1cd408ae2/windows-capture-python/src/lib.rs#L313-L383)、[WGC handler](https://github.com/NiiightmareXD/windows-capture/blob/c7d106448eb9d9b251345c39047711e1cd408ae2/windows-capture-python/src/lib.rs#L588-L624)

另行核查当前依赖源码，固定为 `c7d106448eb9d9b251345c39047711e1cd408ae2`：Rust→Python 桥将帧纹理复制至 staging texture，再 `Map` 成 CPU 可读内存；NumPy `frombuffer` 是对该映射内存的视图，不等于画面从显存直接变成 CUDA tensor。它保留 native owner 维持视图生命周期，且将原生 frame timestamp 传为 Python `timespan`；VisionAI 未利用后者。[native 映射](https://github.com/NiiightmareXD/windows-capture/blob/c7d106448eb9d9b251345c39047711e1cd408ae2/windows-capture-python/src/lib.rs#L440-L486)、[timestamp 桥接](https://github.com/NiiightmareXD/windows-capture/blob/c7d106448eb9d9b251345c39047711e1cd408ae2/windows-capture-python/src/lib.rs#L598-L624)、[NumPy 视图](https://github.com/NiiightmareXD/windows-capture/blob/c7d106448eb9d9b251345c39047711e1cd408ae2/windows-capture-python/windows_capture/__init__.py#L267-L283)

工程推断：该路径仍比“CPU 图像→JPEG→解码→tensor”更直接，但实际速度取决于驱动、窗口大小、映射同步及游戏负载，不能从库名或其宣传 FPS 推出本机端到端延迟。

## 本仓库的问题在哪里

当前在线推理**已经从 `self.pixels` 取得内存 bytes，JPEG writer 已有独立线程**。因此准确的问题是：把压缩图像当成了在线管线内部数据接口，同时将录制完整性和控制健康绑在一起，并非在线 actor 实际等待 JPEG 文件落盘后读取。[policy_runtime.py](../src/fh5/policy_runtime.py)、[policy_writer.py](../src/fh5/policy_writer.py)

代码事实如下：

1. `WindowsColorFrames.capture()` 把 MSS `grab`、BGRX→RGB、缩放到 960×540、JPEG q90 编码连续执行。直到编码结束，帧才 `available`。[live_vision.py](../src/fh5/live_vision.py)
2. `FrozenActor.predict()` 校验压缩 bytes 的 SHA256，`Image.open(BytesIO(...))` 解码，再缩放到模型尺寸；当前资产使用 480×270。缓存最多保留约 32 个解码结果，不能消除新帧首次解码和前面的编码。[policy_actor.py](../src/fh5/policy_actor.py)、[冻结模型验收记录](validation/t08-policy-driving.md)
3. `_frames=Queue(maxsize=2)` 满时丢弃刚来的新帧，`read()` 每次只取一个最旧帧。这限制了队列长度，却优先保留更旧数据。[live_vision.py](../src/fh5/live_vision.py)
4. `_observation()` 按 `delivered_ns` 的历史槽选择，再用 `capture_start_ns` 计算年龄。交付等待被混入时序间隔，不能等同于源画面的等间隔历史。[policy.py](../src/fh5/policy.py)
5. 普通 `frame_queue_drop` 不立即停止，但已接管时凑不齐历史使 observation 不可用，随后以首个原因 `incomplete_image_history` 停止；writer 容量 8，入队失败成为 `image_writer_backpressure`，writer 错误也停止尝试。[policy_runtime.py](../src/fh5/policy_runtime.py)、[policy_writer.py](../src/fh5/policy_writer.py)
6. `_numeric()` 已将每帧 mask 和 age/1000 输入模型。缺少的是正确的源时间采样、容错调度和相应训练分布，不能说目前完全没有图像时间输入。[bc_learning.py](../src/fh5/bc_learning.py)
7. 动作 journal 每次发送仍同步 `write/flush`；原始包记录也逐包 `write/flush`。移走 JPEG 之后，仍需移走这些实时线程上的文件 I/O。[policy_runtime.py](../src/fh5/policy_runtime.py)、[experiment.py](../src/fh5/experiment.py)

### 既有记录的时序证据

对 T08 的已有 `policy.json` 与 `vision.jsonl` 重新计算。仅统计 `status=sent` 的策略决策及其实际引用的去重帧，不能代表所有采集帧，也不包含失败决策的总体尾部。百分位使用排序后线性插值；各阶段的百分位不能相加。原始记录身份、SHA256 和运行限制见 [T08 验收](validation/t08-policy-driving.md)。

| 既有运行 | 实际客户区 | 被引用帧 / 已发送决策 | grab P50 | grab 后至可用 P50 | 可用至交付 P50 | 最新采集起点至发送返回 P50 / P95 | 推理调用墙钟 P50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `t08-bc-assisted-20260930-a` | 3840×2160 | 6 / 4 | 39.89 ms | 26.84 ms | 6.55 ms | 162.51 / 195.91 ms | 31.24 ms |
| `t08-bc-assisted-20260930-b` | 1600×900 | 227 / 119 | 16.40 ms | 11.15 ms | 13.01 ms | 92.42 / 109.21 ms | 30.18 ms |
| `t08-bc-no-reference-20260930-a` | 1600×900 | 229 / 119 | 16.00 ms | 9.59 ms | 12.37 ms | 92.51 / 95.34 ms | 30.79 ms |

口径：`grab=capture_end-capture_start`；后处理为 `available-capture_end`；交付等待为 `delivered-available`；最新画面至发送返回为 `returned_ns-max(selected.capture_start_ns)`；推理墙钟为 `inference_returned_ns-decision_ns`。重算脚本与结果在本地 `.scratch/image-pipeline-research/audit-timing.py`、`.scratch/image-pipeline-research/local-timing.json`，不作为必须提交的机器记录。

这里的后处理包含颜色转换、缩放、JPEG 编码，不能将整段 26.84 ms 全归于 JPEG；推理墙钟也包含调度及 CPU 预处理，不能当作 GPU kernel 时间。`capture_start` 不是 compositor 的原始画面时间，最终指标也只是发送 API 返回，未测游戏真正采用动作的时刻。4K 样本仅 4 个决策，其分位数只有诊断价值。三次运行均未使车辆正常起步，不能将这些数字当作动态驾驶验收。[T08 原始限制](validation/t08-policy-driving.md)

证据支持的判断是：grab、前处理、交付轮询和推理调度均有明显成本；单独修改图片格式、降低保存频率或放宽超时，不能完成用户要求的管线改造。

## NVIDIA API 与 Windows 采集后端的选择

本机已有清单是 Windows 11、i9-13900KF、约 64 GB RAM、GeForce RTX 4090 24 GB；该清单本身不是性能测试。[hardware-snapshot.json](hardware-snapshot.json)

| 候选 | 一手资料确认的能力/限制 | 本项目取舍 |
| --- | --- | --- |
| NVIDIA NvFBC / Capture SDK | 当前 9.0.0 面向 Linux；官方声明 Windows 10 及以上 NvFBC 已弃用，最后支持 1803/build 17134 | 不作为本机 Windows 11 的受支持主线；这不等于断言所有旧版本绝无可能运行 |
| NVIDIA NvIFR | 官方 FAQ 说明 SDK 7.0 弃用、7.1 移除相关定义、文档和样例 | 也不以它作为新的 Windows 实现基础 |
| Windows Graphics Capture | Windows 原生 GPU surface；可按窗口采集，free-threaded frame pool 在内部 worker 发事件 | 保留为参考资料；不作为本次实施主线或必要对照 |
| DXGI Desktop Duplication | Windows 原生桌面复制；NVIDIA 提供 DDA→NVENC 官方示例 | 已选定主后端；核对目标显示器、adapter、客户区裁剪及覆盖物 |
| DXcam | 维护者提供有界环缓冲、最新帧、源时间接口；`video_mode=True` 可重复上一帧填充频率 | 首版包装候选，显式选择 `backend="dxgi"`；关闭录像补帧，向预处理输出数值数组 |
| NVENC | GPU 硬件视频编码，独立于 CUDA 核心的编码引擎 | 适合旁路录像；它不替代采集，不应在 actor 前先编码再解码；见 [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk) |

来源：[NVIDIA Capture SDK](https://developer.nvidia.com/capture-sdk)、[NvFBC Windows 弃用说明，尤其第 4–6 页](https://developer.download.nvidia.com/designworks/capture-sdk/docs/NVFBC_Win10_Deprecation_Tech_Bulletin.pdf)、[NvIFR FAQ Q30](https://developer.download.nvidia.com/designworks/capture-sdk/docs/7.1/NVIDIA-Capture-SDK-FAQ.pdf)、[NVIDIA DDA 编码示例](https://github.com/NVIDIA/video-sdk-samples/tree/master/nvEncDXGIOutputDuplicationSample)、[WGC frame / surface](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.direct3d11captureframe?view=winrt-28000)、[WGC free-threaded API](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.direct3d11captureframepool.createfreethreaded?view=winrt-26100)、[DXcam 官方仓库接口说明](https://github.com/ra1nty/DXcam#frame-buffer)。

**选择依据及讨论的适用范围：**用户提供的 2024-01-31 GStreamer 讨论中，开发者不推荐当时使用 WGC 做显示器捕获，后续回复还涉及跨 GPU 捕获的性能差异。它支持本次选择 DXGI，但不能外推为所有版本的 WGC 窗口捕获都更慢。同一开发者在 2024-08-01 说明，在 d3d12 路径重写 WGC 捕获逻辑后，解决了旧 d3d11 WGC 路径遇到的低帧率问题。因此本项目直接采用 DXGI，并用本机端到端数据验收，不以旧讨论中的 FPS 作为性能承诺。[用户提供的讨论](https://discourse.gstreamer.org/t/d3d11screencapturesrc-dxgi-vs-wgc/912/2)、[后续实现说明](https://discourse.gstreamer.org/t/d3d11screencapturesrc-vs-d3d12screencapturesrc/2080/2)

**实施顺序：**先接入 DXGI 数值采集并建立正确并发结构，与现有 MSS 基线比较。固定 DXcam 版本及实际 API；用目标显示器所属 adapter 建立捕获，再按 FH5 客户区裁剪，不能把桌面区域捕获当成 HWND 独立窗口捕获。初版输出 BGRA 数组，将颜色转换和 resize 留给预处理 worker；保留源时间，关闭 `video_mode` 补帧，并检查窗口移动、尺寸变化、失焦及设备访问失效后的恢复语义。[DXcam 后端、输出与时间接口](https://github.com/ra1nty/DXcam)

若主成本仍是全尺寸 GPU→CPU 映射或 CPU resize，再实现 `DXGI D3D11 texture → CUDA resize/颜色/归一化 → torch tensor`。GPU 方案需要自有纹理池、图形与计算同步、兼容 adapter，以及明确的 buffer 生命周期；不能将 SDK 返回的纹理随手包装为 tensor 就假定可安全复用。迁移是否值得由 P95/P99 新鲜度和游戏帧时间决定，而非仅看采集峰值 FPS。

消除 CPU 往返不等于消除全部拷贝。常见可控实现是 capture surface 先 GPU copy 至兼容的自有 D3D11 texture 池，初始化时注册资源，逐次 map/unmap 并同步，再以 CUDA resize/颜色/归一化写入线性 tensor；不能假设任意 capture surface 均可注册，也不能把 D3D11 texture 直接当作 DLPack tensor。用 DLPack 共享兼容的数值 tensor 时，仍必须遵守生产方存储的所有权与 stream 同步。[CUDA D3D11 互操作 API](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__D3D11.html)、[PyTorch DLPack](https://docs.pytorch.org/docs/stable/dlpack.html)

## 建议的管线契约

```text
DXGI Desktop Duplication capture
        │ raw 数值帧 + 源时间 + epoch + frame_id
        ▼
最新待处理槽（容量 1，覆盖旧的待处理项）
        │
        ▼
常驻 preprocess worker：颜色转换 + 一次 resize
        │ RGB uint8 / 已准备 tensor
        ▼
短历史环：有界、按源时间排序、引用期间不可改写
        │ 非阻塞快照，按时间与新鲜度选图
        ▼
常驻 inference worker：最多 1 个在途任务
        │ decision_id + epoch + source_times + deadline + 数值动作
        ▼
独立 supervisor：验证当前 epoch / 截止时间 / 动作有效期 → send

短历史环 ──有界、非阻塞旁路──> 无损训练数据 / JPEG / NVENC / journal
```

### 1. Capture 只发布数据和时钟

采集回调只获取源时间、窗口/尺寸状态，取得可持续使用的帧引用，或复制进自有缓冲，发布到 latest 槽后返回。它不 resize、不编码、不推理、不写日志文件。只有待处理项被覆盖；被预处理 worker 持有的内容不得改写。若后端资源不能安全保留，就有界复制，不为追求“零拷贝”牺牲所有权正确性。

每帧至少携带 `epoch, frame_id, source_time_ns, capture_received_ns, preprocess_ready_ns, width, height, pixel_format, preprocess_version`。DXGI 源时间使用 `LastPresentTime`，记录 QPC 原始时钟、单位及包装库的转换方式。统一到本机单调时钟后再计算 age，不能直接把不同单位或基准相减；应用接收时间也不能冒充呈现时间。[Microsoft DXGI 帧时间定义](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/ns-dxgi1_2-dxgi_outdupl_frame_info)

DXGI `LastPresentTime=0` 可能只是鼠标更新，不应当作新游戏画面。DXcam `get_latest_frame()` 默认可能阻塞，必须隔离在取帧 worker；使用 `copy=False` 时数据可能被后续采集覆盖，消费者须持有安全副本或具备正确的池引用约束。[DXGI 帧信息](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/ns-dxgi1_2-dxgi_outdupl_frame_info)、[DXcam API](https://github.com/ra1nty/DXcam)

如果后端没有源时间，只能使用采集开始/结束或接收时间作为代理，并明确其不确定性。颜色格式、行 stride、HDR/色彩空间、窗口 resize 必须属于预处理契约，不能只约定 `width × height`。

### 2. 预处理只做一次，历史保留数值数据

常驻 worker 每次取最新尚未处理的帧，完成所需颜色转换与一次目标尺寸 resize，然后发布不可变的 RGB uint8。CPU 路径可使用预分配缓冲；CUDA 路径可在独立 stream 上传、转换，发布时携带 ready event。不得逐决策重新解码、重新缩放已处理帧。

若使用 pinned memory 和 `non_blocking=True`，预先分配有界 pinned 池，DMA 完成前不可复用或改写源槽；在实时路径逐帧调用 `pin_memory()` 未必更快。copy/compute 是否真正重叠仍需测量。[PyTorch 官方说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)

现有权重先保留 480×270；320×180 是独立训练/评估候选，不能因参考项目使用它就偷换已有模型输入。即使尺寸相同，删除原来的“960×540→JPEG→480×270”也改变数值分布，仍须新版本预处理契约和候选评估。

历史目标覆盖至少 0.5 秒，可按 `ceil(预处理频率 × 历史长度)` 加消费者持有余量计算容量，例如 30 Hz 的 16 帧或 60 Hz 的 32 帧。32 张 480×270 RGB uint8 约 11.9 MiB，不需要存数百张大尺寸 float32。实际部署根据频率上限与引用数量选容量，不能仅设置“32”却没有生命周期约束。

原始帧槽、预处理历史和存档各有持有上限。录像队列不能长时间占满 actor 的全部缓冲池；超过旁路配额应丢弃旁路项。工作已超过可用年龄预算时可跳过，不能让旧预处理任务排成长队，也不应在帧率很高时因“有更新帧”而无条件取消每次正在完成的工作。

### 3. 固定间隔选帧，同时传递真实 age/dt

建议首版仍取三帧，名义偏移为 `[200, 100, 0] ms`；这些数值及排列必须写入模型契约。一次决策开始时刻为 `T`：

1. 只快照 `preprocess_ready_ns <= T`、同一 epoch 的帧；绝不等待未来帧补历史。
2. 以快照内最新有效源时间 `F` 为锚，目标时刻为 `F-200 ms, F-100 ms, F`。
3. 对各槽取不晚于目标的最近源帧，并设置最大偏差，例如初始候选 40 ms；或者使用同样有界的最近邻规则。规则固定、版本化，在线与离线完全一致。
4. 要求不同 `frame_id`、严格递增的源时间，不把重复回调或录像补帧当作独立视觉证据。
5. 同时要求最新帧 `T-F` 满足输入年龄预算；再给预期推理和发送保留余量。容差和年龄上限是不同约束。
6. 全部满足才形成观测，向模型提供实际 `age_i=T-source_i`、相邻 `dt_i` 和 mask；若任何必需槽缺失，返回 `SKIP_NOT_READY`。

40 ms、0.5 秒历史及下述频率是待本机验证的初始设计值，不是已测性能。启动时自然需要积累约 200 ms 历史；这段时间保持未接管状态，无需伪造旧帧。

本项目已经有 age/mask 数值字段。第一阶段可保留这些字段并记录 dt，不必无依据扩张模型；若明确增加 dt 或支持任意不规则帧序列，则一起升级模型输入并训练。mask 存在不代表旧权重已学会缺图：当前 actor 本就拒绝不完整历史。时间间隔改变、突发丢帧、捕获延迟随机化与必要的缺图训练须经离线和闭环验证后，才允许“不完整历史仍决策”。

### 4. 推理与控制时钟独立，不积压待算动作

使用常驻单推理 worker，模型加载与预热一次完成。初步目标为 capture 60 Hz、preprocess 30–60 Hz、决策 20 Hz；迁移对照先保持现有 10 Hz，再单独验证频率提升，30 Hz 仅在游戏共享 GPU 负载下的 P99 预算足够时启用。某次时钟到达时 worker 忙，直接跳过该 tick，不创建待推理 FIFO。下一次可工作时重新取得最新快照，而不是计算几百毫秒前排队的动作。

推理结果携带 `decision_id, epoch, selected_frame_ids, source_times, deadline`。发送前重新检查 epoch、最新图像 age、遥测有效性和截止时间。过期结果丢弃。不能因为任务“已经算完”就发送过时动作，也不能简单要求结果完成时必须使用当时绝对最新一帧，否则正常推理也可能永远过不了门槛。

不对任意正在执行的 CUDA kernel 假设可安全强制取消；超时任务仍占该 worker，禁止为它不断新建推理线程。持续超时由监督器使动作失效、停止或隔离 worker，恢复需要明确的重建流程。

### 5. 缺帧跳决策，动作仍有独立失效时间

`SKIP_NOT_READY` 不发送新策略动作，不续租旧动作。最近一次有效动作只保持到已经确定的绝对 `valid_until`；不能每次跳帧又把它延长。暂时仍保留既有 250 ms 硬看门狗，直到另行验收替代值；可在其前配置更早的中立化或已验证的有界制动。监督器独立运行并明确 action owner，不能等推理线程或写盘线程醒来才释放控制。

有效期可按 `min(决策时刻 + hold_limit, 最新源时间 + 容许视觉年龄, 相关遥测失效时间)` 计算，并受既有看门狗上限约束；具体阈值按速度与已有停止行为验证，不能借本次管线改造默默放宽控制范围。

| 事件 | 建议行为 |
| --- | --- |
| 一张帧被覆盖、一次短暂采集失败 | 计数，继续采集；观测满足则推理，否则跳该次 |
| 历史不齐、一次 deadline miss | 跳过，不续租；新观测恢复后继续 |
| JPEG/NVENC writer 拥塞或失败 | 丢旁路帧并标记记录不完整；主路径继续 |
| 推理返回时已过期或 epoch 改变 | 丢结果，绝不下发 |
| 失焦、重开、倒带、车型/尺寸等观测语义改变 | 使旧动作失效，清历史并递增 epoch；按对应状态重新准备 |
| 连续失明、遥测断流、推理挂起超过动作有效期 | 独立监督器归零/执行已验证停止行为，不无限保持旧命令 |

“偶尔不做决策”与“持续失去感知却无限保留旧油门”是不同状态。允许丢帧应减少不必要的停止，同时保持这一区别。

### 6. 存档与学习数据完整性单独管理

JPEG 截图、NVENC 录像、压缩、SHA 和 journal 批量写入均放在有界旁路。它们接收数值图像或引用之后自行编码，actor 不从旁路反向读取。关键动作、epoch、时间、选帧 ID 元数据优先，视频/截图较低优先；队列满不能阻塞控制线程。

为精确重放，CPU 路径保存**实际模型输入所依据的 RGB uint8 数组**及预处理版本、选帧 ID 和时钟信息，可以无损分块压缩或直接数组存储。若 GPU 路径直接生成 float tensor、没有规范 uint8 中间态，则保存实际数值或具备可验证精确重建的输入与处理记录，不能为了存档额外做有损量化。JPEG 是有损格式，JPEG 解码结果不能声称等于当时 actor 看到的数值像素。若 exact model-input 存档丢失，则对应 transition 标记为不可完整重放、不能进入要求完整数据的 replay，也不能用于未经证据支持的晋升评价；这独立于本轮能否继续控制。

旧 JPEG 数据可在离线加载阶段解码一次，再构建版本化训练数组；在线 actor 接口不保留 JPEG 兼容路径。旧 checkpoint 的 provenance 原样保留，新候选记录去编码、采样时钟、resize 和颜色规则的变化。将全部热路径 I/O 移走后，还要测 writer 对 CPU、内存和 GPU 带宽的资源竞争，不能仅因“独立线程”就假定互不影响。

RL 数据还需记录跳过决策后的实际 action hold 时长 `dt`、监督器动作及 owner，不能把不等长真实交互伪装成固定 step。只有时间连续、动作归属明确、所需像素齐全的 transition 才能进入相应训练；缺失记录不能用猜测补齐。

## 如何验证方案而不是追求零丢帧

首要指标是 **动作发送时最新输入源帧的年龄**：`Age_at_action = action_send_return_ns - newest_source_time_ns`，同时记录 `decision_to_action`、命令间隔和有效命令占比。不能只报采集 FPS 或 GPU 推理均值。Windows/Python/共享 GPU 属于软实时环境；有界队列和 deadline 保证不会主动消费无限积压，不构成操作系统级硬实时承诺。

首轮候选性能目标可设为新源时间口径的 `Age_at_action P95 ≤ 75 ms、P99 ≤ 120 ms`，同时报告最长连续 skip、最近动作实际持续时间和有效决策率；这些是待验收目标，不是已有结果或允许忽略动作租期的理由。达不到目标应报告实际决策频率和 skip、再定位瓶颈，不以延长过期时间掩盖。T08 后两轮约 15 秒各发送 119 次策略命令，约 7.9 Hz，可作为有效更新率的历史背景。新后端原生呈现时间与旧 `capture_start` 口径不同，正式 A/B 应补齐统一时间线，不能直接以表中旧数值计算精确加速比。

| 类别 | 必须记录或检查 |
| --- | --- |
| 分阶段时间 | capture/原生等待、GPU→CPU、resize/颜色、队列驻留、H2D、推理、结果返回、send；P50/P95/P99/max |
| 新鲜度 | 源帧至动作年龄、所选时间误差、实际 dt、ready age、最长命令间隔 |
| 负载与有界性 | 活跃 pool 槽、inflight 数、内存/显存峰值、CPU占用、游戏帧时间 |
| 丢失语义 | capture/preprocess/sampling/archive 各自 drop；skip 原因；过期结果；不可重放 transition |
| 正确性 | 色序/stride/HDR、源时钟转换、被引用槽不可覆盖、epoch 边界、不复用重复源帧 |

对照实验应保持实际客户区、游戏场景、渲染设置、模型、功耗状态及记录负载一致，分别测：当前 MSS+JPEG；MSS 数值输入；DXGI 数值输入；仅在仍有瓶颈时加入 GPU 预处理。MSS 数值输入仅作为拆分成本的诊断条件；WGC 对照不是本次交付前置。这样才能拆开“去 JPEG”“换后端”“缩小源窗口”和“减小模型”各自的收益。预热与冷启动分开统计，GPU kernel 用 GPU event 计时，端到端用统一单调时钟。

故障注入必须覆盖：5%/10% 随机丢帧、100 ms 突发缺帧、300 ms 断流、writer 卡 1 秒、推理超过 deadline、重复旧帧、槽复用压力、暂停/重开/倒带/尺寸变化。验收目标分别是及时跳过并恢复、存档降级不堵控制、过期结果不发送、动作到期释放、无跨 epoch 历史；不是要求这些测试中 drop 始终为零。

推荐实施分三步，每步可独立比较：

1. **修正数据与调度接口。** 数值 actor、三阶段分离、latest 覆盖旧待处理项、常驻推理、有界异步存档、跳帧与动作租期。先保持当前模型尺寸和决策频率，验证软件行为与现有录制重放。
2. **接入 DXGI 并做被动 A/B。** 固定依赖版本，显式选择 DXGI，与 MSS 基线比较；验证 adapter/显示器匹配、源时间、图像一致性、真实负载下的年龄分位数和资源占用，再决定决策频率。
3. **按证据优化 GPU 与训练。** 当映射/CPU resize 仍占主导时做 D3D11→CUDA；输入分布和时间契约已变化时重建训练数据并验证新候选，再进入有界实机驾驶验收。

本调研不修改 [ADR 0004](adr/0004-multimodal-learning-before-geometry.md) / [ADR 0005](adr/0005-visual-navigation-optional-reference.md) 的多模态学习方向。后续实现应更新图像观测与模型契约；当前规格中的“缺图即终止”“必须完整存档”等约束若与用户新要求冲突，应明确版本化替换，不能一边宣称容错一边继续由 writer 满队列终止驾驶。
