# 数值输入与精确回放

T31 / GitHub #33 增加离线数值路径：准备 → 冻结策略 → 异步存档 → 精确回放。统一通过 `run_experiment`，不创建虚拟手柄、不启动 FH5。当前旧 BC 权重只用于显式兼容诊断；DXGI、显式 Δt 模型和容错实机决策分别由 #34、#35、#36 接续。

## 命令

需要 `uv sync --locked --extra learning`。以下使用已有本地资产，输出目录/报告名必须未存在。

```powershell
uv run --locked fh5 numeric-prepare --model runs/t28-bc-initial --dataset runs/t27-dataset-reviewed/dataset.json --output runs/numeric-prepared --max-decisions 200
uv run --locked fh5 numeric-infer runs/numeric-prepared --model runs/t28-bc-initial --output runs/numeric-inference --legacy-diagnostic --archive-capacity 256 --archive-mib 256
uv run --locked fh5 numeric-replay runs/numeric-inference --model runs/t28-bc-initial --report runs/numeric-replayed.html --legacy-diagnostic
```

准备阶段核对模型绑定的数据集、原遥测、观测与压缩图像依据；每个选中的源图像只解码一次并缩放，生成 RGB 数值资产。原文件不改动。可用 `--view reference_assisted` 检查有参考视图，默认 `no_reference`。

`numeric-infer` 默认 CPU，可选 `--device cuda`。加载权重和准备 manifest 发生在执行前；源工作线程预读数值资产，决策线程只接收拥有独立存储的数值帧并调用模型。旧 JPEG 不进入这个阶段。`--legacy-diagnostic` 必须显式给出；这不表示权重适用于新 DXGI 分布，也不证明实时性能。

## 像素与时间契约

- `PixelContract` v1：RGB uint8、HWC 连续布局、完整画面、Pillow bilinear v1、`float32 / 255`，历史由旧到新。实际尺寸和历史偏移写入记录；真实迁移样例为 480×270、200/100/0 ms。
- `NumericFrame` 对传入 buffer 做不可变快照，向 actor 提供只读三维 `memoryview`，无需图像解码。Torch 适配器从数值缓冲建立 CHW tensor，再执行冻结模型。
- 每帧保存 `frame_id`、`epoch`、源时间、时间质量与不确定性、接收时间、预处理可用时间、预处理版本及源布局。三个时刻使用同一单调时钟域且依次不减；可用时间不得晚于决策。相邻帧源时间严格前进，禁止重复图像、跨 epoch 或混合布局。
- 生产者负责在暂停、倒带、重开、相机/HUD 或尺寸语义变化时递增 epoch。接口检查提交的边界，不从像素猜测游戏状态。现有离线导入沿用原观测的前向片段边界。
- 历史数据保留 `capture_start_proxy` 与未知不确定性；`legacy_encoded_delivery_proxy` 表示历史编码交付时刻。它不是实测数值预处理延迟，更不是 DXGI 呈现时间。
- 模型继续使用原 BC 的本车、因果动作、图像年龄及可选航点特征。显式帧间 Δt 待 #35；不得把旧模型年龄特征等同于已实现的新 Δt 方案。

## 记录和资源

`NumericInfer` 接受显式 `Iterable[NumericDecision]` 与冻结 `NumericActor`；直接数值生产者不需要任何图像文件。`PreparedNumericSource` 是离线适配器，它在独立线程读取 `.rgb`，不是新的在线采集实现。

输出包含 `numeric-run.json`、`inputs/*.json`、按内容哈希去重的 `pixels/*.rgb`、独立 `previews/*.png` 及报告。数值像素、特征和预测进入带哈希的决策记录。哈希、原始数值写盘和展示图编码都在存档线程中；展示图不作为预测输入。

存档默认最多持有 8 个任务、32 MiB 数值引用（含正在写入者），提交不等待磁盘。配置最多 256 个任务、1 GiB；超过任一界限会跳过该次存档，保留预测并标记 `exact_replay_available=false`。离线批量推理可瞬间超过实时到达速率，示例为此显式扩大预算；这不是在线推荐值。源预读队列为 2，缓存默认 16 MiB；旧 BC tensor 缓存最多 32 帧，结束后清空。各预算独立，不等于进程总内存。

停止时关闭源并回收缓存、排空存档；工作线程有两秒退出预算。若底层 I/O 挂起，结果明确记录资源未释放与不可回放，不能算完整运行。异常保留已完成决策；未保存的数据不以邻帧补造。

## 查看与验收

报告按决策同时切换历史图片、来源/时间、actor 特征、预测和记录完整性，支持播放及拖动；图片使用解码完成后再替换的展示缓存。报告依赖本地 `previews/`，分享时应连同数值包保留。浏览器禁止本地文件时，页面交互验收仍待完成，不能以静态 HTML 生成成功代替。

回放逐项检查数值资产哈希、元数据与汇总一致性、像素契约、输入特征和预测容差（默认 `1e-6`）。展示图缺失/改变不影响数值预测；数值资产缺失、被篡改或越过记录目录会产生 `replay_errors`。这是完整性核对，不是防恶意重写全部 manifest 的签名方案。

旧 `record/replay`、`vision/observe`、BC 工具及旧 `policy` 后端继续保留。**旧 `policy --live` 尚未切换到数值管线**，不能据本票的离线结果直接恢复实机驾驶。验证证据见 [T31 记录](validation/t31-numeric-images.md)。
