# DXGI 数值采集基础切片

对应 #34，目前提供采集/预处理解耦、源时间、历史选择及诊断入口。**本票未验收完成**：尚缺原生画面动态实测、同尺寸 MSS 对照及页面交互；不会据软件测试关闭 #34 或宣称已满足驾驶时效。

## 入口

Windows 可选依赖为 `capture`，锁定 `dxcam==0.3.0`、`Pillow==12.3.0`。开发者先按 `uv.lock` 安装对应 extra，不改变控制驱动。

```powershell
uv sync --locked --extra capture
uv run --locked fh5 capture-dxgi --config configs/capture-dxgi.example.json --output runs/dxgi-001
uv run --locked fh5 capture-dxgi --config configs/capture-dxgi.example.json --output runs/dxgi-001 --seconds 30 --live
```

第一条运行命令仅校验配置；`--live` 被动读取前台 FH5 客户区和 UDP 5300，不连接虚拟手柄，F8 停止。没有前台 FH5 时不创建截图源，不捕获其他应用。输出目录必须不存在。秒数是整次探测时长，包含等待前台时间；没有新帧会返回非零状态，不把等待称作采集成功。

优先要求实测 3840×2160 客户区，输出 480×270；尺寸不匹配记录未就绪，不自动降档。`device_idx/output_idx` 必须对应游戏所在显示器，整个客户区须位于该输出内。当前显式拒绝旋转显示器。示例中的游戏内部渲染、上采样、HDR、HUD 和 FOV 为 **unverified**，实测前确认并冻结新的条件 ID；桌面尺寸不能代替这些证据。

## 内存、线程与时间

采集线程调用 DXGI one-shot `grab(copy=True,new_frame_only=True)`，只复制 BGRA 数值；不启动 DXcam 自带采集线程或录像补帧。容量为一的待处理槽覆盖旧待处理项。独立常驻预处理线程完成正在处理的帧，再取最新项；一次 BGRX→RGB、一次 Pillow bilinear resize，形成不可变数值历史。历史最多 64 帧（默认 32），输出最多 640×360，单个原始裁剪最多 64 MiB。

观察入口按最新已完成源帧为锚，选择 `[200,100,0] ms` 目标之前的最近帧，偏差上限 40 ms、最新图像年龄上限 100 ms。缺少历史、重复源帧、过旧图像或过大的采样偏差返回未就绪；不等待未来画面。参数是待实测候选，不代表已有性能保证。

QPC 整数 ticks 先减参考值再映射到 `perf_counter_ns`，记录校准采样跨度和一个 QPC tick 的不确定性。帧同时保存原生呈现时间、接收时间、预处理完成时间，报告 Δt 和 age。窗口/裁剪/DPI/输出变化或捕获错误切开 epoch，旧预处理即使完成也不得进入新历史。

锁定版本有两个需要绕开的行为：

- DXcam 的时间字段在 `LastPresentTime=0` 时可退回鼠标更新时间。本适配层读取同一次 `AcquireNextFrame` 返回的原生 frame info，零呈现时间及非前向时间不会成为新画面。
- DXcam 内置显示恢复持续重试。本适配层把该入口改为返回访问失效，由外层最多三次关闭/重建；本机调用挂起仍只能报告隔离线程与未释放状态，不能保证 Python 能取消原生调用。

这些私有 SDK 接点仅适用于核对过的 0.3.0；版本不同明确拒绝，升级必须重审。审计 wheel 为 `dxcam-0.3.0-cp312-cp312-win_amd64.whl`，SHA-256 `ab75b16529b8bd7337dfcb663525c338233a24b4c5b76193ced82a70e16edf49`。来源：[DXcam 0.3.0](https://pypi.org/project/dxcam/0.3.0/)、[原生 frame info 定义](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/ns-dxgi1_2-dxgi_outdupl_frame_info)。

## 记录与证据边界

观察快照提交有界异步数值存档，队列 8、持有上限 32 MiB；接受的引用字节预算默认 512 MiB（保守重复计数），超出后停止存档该观察，不反压采集。PNG 仅作显示副本；预处理/选帧不读写图像文件。报告保留每个观察的存档资格，缺失记录不成为完整训练数据。

`capture.json/report.html` 包含选帧、原始时钟、阶段耗时 P50/P95/P99/max、覆盖计数、历史容量与资源释放结果。遥测目前仅用于被动活动诊断，明确标为接收轮询，**不作为同步示范动作或帧物理时刻证据**。移动观察需要新鲜活动遥测且速度超过 1 m/s；这一计数本身也不是正式实机验收。

`CaptureReplay` 经 `run_experiment` 注入原始 BGRA、QPC 映射、接收和处理耗时，使用同一历史状态机检查积压、重复呈现与 epoch。其耗时标为 simulated，不能与实机数据一起报告为性能。

仍需推进的软件及实机验证：原生 SDK 桥的独立核验；完善源帧率/缺口、资源峰值和游戏帧时间报告；同实际客户区的 MSS 数值/MSS+JPEG 诊断对照；少量 4K 原图对照样本；动态只读短测与播放/拖动检查。当前不训练模型、不发送动作，也未接入 #36 的容错策略运行器。
