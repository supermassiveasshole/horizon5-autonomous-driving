# 旧模型与数值包的回放和迁移

本页维护已有旧模型、编码图像录制和数值诊断包的读取与迁移。可按需阅读[旧 BC 模型](#bc)、[旧驾驶入口迁移](#policy-driving)或[数值包回放](#numeric-images)。以下离线回放入口仍可使用；新实验从[持续采集](../guides/capture/continuous-collection.md)、[数据准备](../guides/bc/collection-datasets.md)和[当前 BC 训练](../guides/bc/training.md)开始。原始数据与历史结论保留。

<a id="bc"></a>

## 旧版 BC 模型回放与迁移

旧 `bc-train` / `BCTrain` 及其优化循环已退役。新训练统一使用[数值 Δt BC](../guides/bc/training.md#temporal-bc)，采集期间使用[单配置后台训练](../guides/bc/training.md#learning-schedule)。旧 `bc-replay` / `BCReplay` 继续读取 v1 权重和历史示范，检查契约并生成诊断，不连接游戏、不发送输入，也不修改旧模型。

### 运行

```powershell
uv sync --locked --extra learning
uv run --locked --extra learning fh5 bc-replay --model runs/bc-first --dataset runs/t27-dataset-reviewed/dataset.json --report runs/bc-reload.html --device cuda
```

回放使用新报告路径，保留旧模型、数据和源录制。检查使用 `uv run --locked --extra learning pytest tests/test_bc.py`，未安装 PyTorch 的基础环境会跳过模型测试。普通遥测录制/回放入口不导入 PyTorch。

### 迁移新训练

历史示范使用 `temporal-prepare` 导入，再用 `temporal-train` 训练新候选；新数值采集使用 `collection-bc-prepare` 准备，再用 `temporal-train` 或 `collection-bc-train`。旧 `bc-train` 调用只返回迁移说明，旧训练示例已移除，不静默套用旧配置。

这是数据导入后重新训练，不是旧权重转换或旧实验的精确重跑。目标时间历史至少需要两帧；只有单帧可用或不能满足因果时间契约的片段不能构成 Δt 训练输入。导入保留来源、时间不确定性和排除原因；旧 `holdout` 映射为已使用的 `development`，不冒充新的最终留出。历史来源模型也不能据此取得新采集或实机驾驶资格。旧单帧模型及失败观测的诊断仍由本页回放入口支持。

### 数据与输入契约

回放使用原始录制、绑定清单和嵌入的质量审阅重新导出样本，与提供的数据集逐项比较；不直接信任可编辑的 `bc_eligible`、历史动作或 split 字段。图片按哈希校验，保留按独立录制划分。原始文件、审阅及配置仍须保留；哈希核验不是对恶意共同重写所有文件的认证。

Actor 输入仅来自 v2 actor 字段：完整 RGB 历史、本车速度/车体速度/角速度、图像与遥测年龄、先前双轴动作及年龄/掩码、可选本车坐标参考航点及掩码。绝对位置、回合/目的地 ID、全局进度、标签、未来航点和质量审阅均不进入网络。完整图像历史或本车状态缺失时不预测，保留空值；已知失败但观测完整的片段可产生诊断预测，不能成为正确 BC 标签。

数值预处理使用固定物理尺度：速度与车体速度除以 100，角速度除以 5，毫秒年龄除以 1000，参考米数除以 100；掩码保留，缺失动作/航点填零且掩码为假。没有从留出数据拟合统计量。RGB 全幅双线性缩放后转 float32/255；不裁掉导航/小地图，也不假定追尾相机外参固定。

### 旧网络及历史训练方式

以下描述既有 v1 模型的生成方式，当前代码不再执行这条训练路径。

版本 `rgb-history-conv4-64-state64-fusion128-tanh2-v1`：每帧共享四层卷积（16/24/32/48 通道，步幅 2，ReLU），保留 3×5 空间池化特征，投影到 64 维；按历史顺序拼接，与 64 维数值分支融合，128 维隐层输出 tanh 双轴。编码器可训练，完整参数保存在 actor 内，无冻结 latent 缓存。采用 PyTorch 随机初始化，不下载外部预训练权重。

每个抽到的可信训练样本以有参考/无参考两种视图成对进入批次，实际占比各 50%；batch_size 必须为偶数。同一图像缓存 uint8，仅当前小批量转换浮点。Adam（默认 betas/eps，无 weight decay）、两轴等权 MSE；当前不实现未来轨迹辅助头，明确记录关闭、权重零。固定 seed、更新次数与最终一步权重，不使用留出选择超参数或检查点，不做早停。确定性算法不能承诺不同硬件/PyTorch 版本逐位一致。

动作含义保持 `xinput-lx-rt-lt-v1`：转向 [-1,1]；纵向正值为 RT，负值为 LT 服务刹车，非手刹。输出只是策略请求，尚不代表限幅、下发或游戏采用。

### 产物与解释

- `actor.pt`：actor 全部参数（包含 encoder）及绑定的架构/配置/输入/预处理元数据；只用 `weights_only=True` 加载，不序列化可执行模型对象。
- `model.json`：文件哈希、架构、来源、固定条件、训练配置、输入历史、掩码课程与训练诊断。元数据须与权重内绑定值一致，版本/形状/有限值不符时报错。
- `report.html/json`：全部样本的预测、标签、质量与原因；分别报告 train/holdout、有/无参考的双轴 MAE、RMSE、P90、录制/5 秒窗口/速度/转向/踏板分层，以及复制上一输入的诊断基线。空分层没有成绩。

历史图像梯度、编码器参数变化和固定数值输入下黑图干预只说明视觉参与计算/响应，不证明理解导航或带来驾驶收益。只有后续真实重复驾驶能回答这个问题。当前 SAC、冻结评估和实时驾驶使用数值时间模型；旧 v1 权重只能明确用于离线诊断，不包含合格 critic、优化器恢复状态或自动驾驶验收。

本轮实际结果见 [T27 验证](../validation/t27-bc.md)。独立录制留出用于本轮初始化诊断；被查看后不能继续作为后续选型的未见最终评估。

实现采用 PyTorch 官方建议的 [state_dict 保存及 weights_only 加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)，确定性限制参见 [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

<a id="policy-driving"></a>

## 旧模型驾驶入口迁移

旧 `policy` 在线入口及 `PolicyDrive` 已退役；调用 `fh5 policy` 只返回迁移错误，不加载模型或连接设备。当前模型驾驶统一使用 [`fh5 realtime-drive`](../guides/runtime/realtime-decisions.md#有界驾驶命令与条件绑定)。

旧 v1 BC 模型和 `policy` JSON 不会自动转发为新配置。新入口使用来自连续数值采集、符合当前运行条件的[数值 Δt BC（temporal BC）模型](../guides/bc/training.md#temporal-bc)，按 [`configs/realtime-drive.example.json`](../../configs/realtime-drive.example.json) 填写采集配置、候选模型、独立任务和只读运行依据。默认只校验；准备好所需资产并满足条件后，显式 `--live` 才连接控制。具体条件见数值驾驶说明。

当前数值驾驶使用无参考策略，独立任务路线只用于起点、范围和结束核验，不作为策略航点输入；它不等价支持旧 `required` / `optional` 参考模式。停止时解除输入，不保证车辆已经刹停，也不沿用旧路径的终点制动流程。实机驾驶仍待验收。

已保存的旧录制继续支持离线读取，无需游戏，也不会重新发送输入：

```powershell
uv run --locked fh5 replay runs/policy-001 --report runs/policy-001/replay.html
```

使用新的报告文件名。旧录制、模型与[历史验收记录](../validation/t08-policy-driving.md)保留原结论，不因入口退役或回放成功而成为新路径的驾驶证据。

<a id="numeric-images"></a>

## 数值输入与精确回放

本文说明数值输入契约及已有 T31 / #33 离线包的读取。`numeric-prepare` / `LegacyNumericImport` 的旧模型绑定导入实现已退役；新历史训练数据统一用 [temporal-prepare](../guides/bc/training.md#temporal-bc)，新数值采集用 [collection-bc-prepare](../guides/bc/collection-datasets.md)。二者均不要求先创建一个旧 BC 模型。

已有 `prepared.json` 包继续通过 `numeric-infer` 读取，已有 `numeric-run.json` 继续精确回放。原始数据和旧权重不改写；仅有原始单帧数据时仍可用 [bc-replay](#bc) 诊断旧模型，不能承诺转换为 Δt 训练历史。这不是旧导入的同格式替换，不再创建新的单视图、模型绑定诊断包。

### 命令

需要 `uv sync --locked --extra learning`。以下读取已经保存的准备包，输出目录/报告名必须未存在。

```powershell
uv run --locked fh5 numeric-infer runs/numeric-prepared --model runs/t28-bc-initial --output runs/numeric-inference --legacy-diagnostic --archive-capacity 256 --archive-mib 256
uv run --locked fh5 numeric-replay runs/numeric-inference --model runs/t28-bc-initial --report runs/numeric-replayed.html --legacy-diagnostic
```

包中已固定视图、源时间和数值资产。示范中的当前动作、标签时间和未来监督项作为 `supervision` 证据保存，从不传入 actor。读取时继续核对像素哈希及时间/布局契约；原有数据的准备过程与核验证据见 [T31 历史记录](../validation/t31-numeric-images.md)。

`numeric-infer` 默认 CPU，可选 `--device cuda`。加载权重和准备 manifest 发生在执行前；源工作线程预读数值资产，决策线程只接收拥有独立存储的数值帧并调用模型。旧 JPEG 不进入这个阶段。`--legacy-diagnostic` 必须显式给出；这不表示权重适用于新 DXGI 分布，也不证明实时性能。

### 像素与时间契约

- `PixelContract` v1：RGB uint8、HWC 连续布局、完整画面、Pillow bilinear v1、`float32 / 255`，历史由旧到新。实际尺寸和历史偏移写入记录；真实迁移样例为 480×270、200/100/0 ms。
- `NumericFrame` 对传入 buffer 做不可变快照，向 actor 提供只读三维 `memoryview`，无需图像解码。Torch 适配器从数值缓冲建立 CHW tensor，再执行冻结模型。
- 每帧保存 `frame_id`、`epoch`、源时间、时间质量与不确定性、接收时间、预处理可用时间、预处理版本及源布局。三个时刻使用同一单调时钟域且依次不减；可用时间不得晚于决策。相邻帧源时间严格前进，禁止重复图像、跨 epoch 或混合布局。
- 生产者负责在暂停、倒带、重开、相机/HUD 或尺寸语义变化时递增 epoch。接口检查提交的边界，不从像素猜测游戏状态。现有离线导入沿用原观测的前向片段边界。
- 历史数据保留 `capture_start_proxy` 与未知不确定性；`legacy_encoded_delivery_proxy` 表示历史编码交付时刻。它不是实测数值预处理延迟，更不是 DXGI 呈现时间。
- v1 模型使用本车、因果动作、图像年龄及可选航点特征。当前 v2 模型的显式帧间 Δt 见[时间契约](../guides/bc/training.md#时间和模型契约)；旧模型的年龄特征不等同于 Δt 模型。

### 记录和资源

`NumericInfer` 接受显式 `Iterable[NumericDecision]` 与冻结 `NumericActor`；直接数值生产者不需要任何图像文件。`PreparedNumericSource` 是离线适配器，它在独立线程读取 `.rgb`，不是新的在线采集实现。

输出包含 `numeric-run.json`、`inputs/*.json`、按内容哈希去重的 `pixels/*.rgb`、独立 `previews/*.png` 及报告。数值像素、特征和预测进入带哈希的决策记录。哈希、原始数值写盘和展示图编码都在存档线程中；展示图不作为预测输入。

存档默认最多持有 8 个任务、32 MiB 数值引用（含正在写入者），提交不等待磁盘。配置最多 256 个任务、1 GiB；超过任一界限会跳过该次存档，保留预测并标记 `exact_replay_available=false`。离线批量推理可瞬间超过实时到达速率，示例为此显式扩大预算；这不是在线推荐值。源预读队列为 2，缓存默认 16 MiB；旧 BC tensor 缓存最多 32 帧，结束后清空。各预算独立，不等于进程总内存。

停止时关闭源并回收缓存、排空存档；工作线程有两秒退出预算。若底层 I/O 挂起，结果明确记录资源未释放与不可回放，不能算完整运行。异常保留已完成决策；未保存的数据不以邻帧补造。

### 查看与验收

报告按决策同时切换历史图片、来源/时间、actor 特征、预测和记录完整性，支持播放及拖动；图片使用解码完成后再替换的展示缓存。报告依赖本地 `previews/`，分享时应连同数值包保留。静态 HTML 生成不能代替交互验收；T31 指定报告已于 2026-10-03 由用户确认播放、拖动和画面/数据联动正常，见[人工验收记录](../validation/t31-numeric-images.md#人工页面交互验收2026-10-03)。

回放逐项检查数值资产哈希、元数据与汇总一致性、像素契约、输入特征和预测容差（默认 `1e-6`）。展示图缺失/改变不影响数值预测；数值资产缺失、被篡改或越过记录目录会产生 `replay_errors`。这是完整性核对，不是防恶意重写全部 manifest 的签名方案。

当前在线入口是 [realtime-drive](../guides/runtime/realtime-decisions.md)，旧 `policy` 在线实现已退役。离线数值结果不构成实机驾驶资格；历史证据见 [T31 记录](../validation/t31-numeric-images.md)。
