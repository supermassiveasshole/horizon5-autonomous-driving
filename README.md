# FH5 歌利亚自主驾驶与学习实验

使用用户固定调校的兰博基尼第六元素，从遥测与已知路线起步，建立能自行采样、训练、评估并改进驾驶策略的实验系统。首版聚焦无对抗歌利亚，后续探索对手竞速和路线泛化。

当前实现：T01 遥测录制与离线回放已获得真实驾驶记录；T02 已在用户第六元素 X 999 上完成低速起步、左右转向、刹停及 F8/失焦/暂停释放验证。峰值约 9.58 km/h，尚未实现路线跟随或自主驾驶。见 [T02 控制说明](docs/control.md) 与 [实机验证记录](docs/validation/t02-control.md)。七天是首轮探索窗口，按可运行里程碑推进。

T03 开发中：[赛事生命周期](docs/events.md) 已接入 Windows 菜单适配器和 `fh5 event`。空场蓝图 `458 574 769` 的程序探测完成三次静止起跑核验及两次连续重开，约 43 秒，全程零驾驶输入；固定环境与持续采样尚未验收，见 [T03 实机记录](docs/validation/t03-event.md)。人工整圈也已保存，但因使用倒带不计入有效基线。`configs/goliath-458574769.json` 是录制快照；菜单探测另需本地页面模板及配置。

当前实验条件更新为蓝图 `105 657 219`：干燥季节、晴朗、固定早晨，配置为 `configs/goliath-fixed-v2.json`。已完成一次赛途中暂停后自动重开，从准备到起点核验约 12.91 秒；旧蓝图记录保留原条件。持续无人值守验收仍未完成。

T04 局部数据验收完成：[局部路线工具](docs/routes.md) 可从连续记录导出参考轨迹，独立保存走廊与检查点依据，并回放连续定位与已确认进度。已保留约 801 米历史参考，拱门后约 19.55 米的保守内部区域 v2 已经独立实测通过：653 个连续样本确认全段，窗口峰值约 9.97 km/h。`fh5 route-check` 保留完整记录、失败与版本依据；受限速度采集不等于学习策略驾驶，局部尝试有效性现由 #7 实现，见 [T04 数据记录](docs/validation/t04-route.md)。下一步按已批准的[多模态学习规格](docs/multimodal-learning-spec.md)推进：含道路/导航的真实截图历史、本车状态、因果动作历史及可选历史航点进入 BC→SAC；先做可信同步示范和离线 BC，训练与评估实际覆盖无参考条件，见[视觉导航增量](docs/visual-navigation-spec.md)。当前 #26 的 v1 保留必选参考回放，#31 已实现可选/禁用参考的 v2 观测，已完成 #28 离线 BC 初始化但未通过实机驾驶验收。精确分割/米制道路保留为可选研究，实机控制和成绩有效性另行验收。

T06 / #7 已加入 `fh5 attempt-review`：完整尝试、前向片段和恢复排除区分别留存，独立证据决定局部结论，正式全程仍单独待核验。支持已有记录离线审核，不发送控制；使用方式见 [局部尝试说明](docs/attempts.md)，真实覆盖与限制见 [T06 验证](docs/validation/t06-attempts.md)。

T09 / #10 新增[冻结评估与全部尝试统计](docs/evaluation.md)：`fh5 evaluation-prepare` 固定 BC 或 SAC、任务和批次条件，`fh5 evaluation-review` 汇总完整录制中的五类结果及未开始计划，并核对数值执行证据。已有[冻结 SAC 的合成重复运行与回放](docs/validation/t09-sac-evaluation.md)，独立验证决策时和发送前的命令上下文。真实重复驾驶与独立最终验收仍待完成，不据此晋升版本。

T13 / #14 的[候选比较](docs/candidate-selection.md)可从原始证据重新审核两个冻结开发批次，先有效性、后可靠性与用时，逐参考条件保留激进候选建议。[持久候选版本](docs/candidate-store.md)通过完整归档分别保留合成默认版本、探索进度、激进候选和回退历史，支持跨进程读取及过期写入保护。实际 FH5 自动晋升和独立驾驶资格仍待完成。

## 安装与运行

T07 / #8 新增[物理时间奖励与终止结算](docs/rewards.md)：`fh5 reward-replay` 重算历史局部片段，`fh5 reward-audit` 生成完整合成反例与回报排序。独立有效性、奖励标签和正式成绩保持各自结论；未启动 SAC 或新的实机驾驶。

T31 / #33 已加入独立的[数值图像输入与精确回放](docs/numeric-images.md)：旧图像仅在离线准备时解码，冻结策略接收数值 RGB，数值存档在后台执行。真实历史数据已有 200 个观测精确回放；页面交互仍待验收。参见 [T31 验证](docs/validation/t31-numeric-images.md)。

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。在仓库根目录运行 PowerShell：

```powershell
uv sync --locked
New-Item -ItemType Directory -Force runs | Out-Null
Copy-Item configs/recording.example.json runs/manual-config.json
```

编辑 `runs/manual-config.json`，填写本次车辆、调校、辅助、赛事和环境；不确定的项目保留 `unverified`。示例的 Sesto Elemento 来自用户选车，具体版本尚未核实。

在 FH5 的 **设置 → HUD 与游戏（HUD and Gameplay）** 中开启 **Data Out**，目标 IP 设置为 `127.0.0.1`，端口设置为 `5300`，进入可手动驾驶的场景：

```powershell
uv run --locked fh5 record --config runs/manual-config.json --output runs/manual-001 --seconds 60
```

看到 `listening` 后开始驾驶。满 60 秒自动结束，或在终端按 **Ctrl+C** 提前停止并保存。打开 `runs/manual-001/report.html` 查看轨迹、车速、播放滑块、异常及快照；页面离线可用。输出目录必须是新目录，避免覆盖实验。

不运行游戏也能重新解析和回放已有记录：

```powershell
uv run --locked fh5 replay runs/manual-001 --report runs/manual-001/replay.html
```

每次回放使用新的报告文件名。接收了 UDP 并不自动证明来源是 FH5；实机核验步骤、字段定义和限制见 [录制与回放说明](docs/recording.md)。

## 彩色观测

T23 已接入冻结 SegFormer 的离线道路候选与独立人工核验流程，使用 `fh5 perceive` / `fh5 perception-replay`。输出仍是像素估计，未获道路精度或驾驶验收；安装、协议与标签说明见 [像素道路估计](docs/perception.md)。

彩色画面与遥测可通过 `fh5 vision` 一起记录，不发送游戏输入：

```powershell
uv sync --locked --extra events
uv run --locked fh5 vision --config configs/goliath-fixed-v2.json --output runs/vision-001 --camera chase-far --seconds 60
```

保持追尾远视角；F8 或输出目录的 `STOP` 文件提前结束。`report.html` 提供彩色回放、当时可用遥测及故障诊断。参数、数据格式和限制见 [彩色观测说明](docs/vision-recording.md)。

T25 / GitHub #26 可将已有 RGB、遥测和独立历史路线构造成因果观测回放：

```powershell
uv run --locked fh5 observe runs/vision-001 --config configs/observations.example.json --route runs/reference/route.json --report runs/observations-001/report.html
```

报告展示图像历史、年龄/缺失掩码、本车状态和局部航点；JSON 保留可重读的原图依赖。新采集可记录真实检查时刻，旧采集明确标记为重建时钟。此阶段不控制车辆，见 [因果观测说明](docs/observations.md)。

T29 / #31 新增观测 v2，可省略历史参考，支持 `required/optional/disabled` 三种模式；独立评测路线不进入策略输入。旧录制的动作历史明确缺失，不伪造中性动作：

```powershell
uv run --locked fh5 observe runs/vision-001 --config configs/observations-navigation.example.json --report runs/navigation-001/report.html
```

这证明无参考观测可构造，还不是无参考驾驶能力；格式、动作历史导入及独立任务证据见 [视觉导航观测](docs/navigation-observations.md)。

## 人工同步示范

#37 的[持续被动采集](docs/continuous-collection.md)只需准备一次冻结安装，之后用 `fh5 collection-start runs/collector --output runs/collection-001 --live` 启动；下一轮换输出目录即可，不重复安装依赖。状态查询、停止及学习调度指向本轮目录，旧记录保留。原生 DXGI/UDP/XInput 组合使用数值像素；后台生命周期已用合成输入验证，真实多次驾驶与 4K 性能仍待验收。

#38 的[数据快照](docs/collection-datasets.md)已接入封存来源筛选、关联尝试分组、数值 Δt BC 训练与冻结留出评估；采集继续追加时固定选择不变。[采集优先的学习调度](docs/learning-schedule.md)已完成合成进程验证；真实新数据、4K 游戏负载与驾驶效果仍待验收。

后台训练使用 `fh5 collection-bc-train --config configs/collection-learning.example.json --output runs/scheduled-candidate-001`。v2 配置将训练参数、采集绑定和资源预算放在同一文件，不再手工维护第二份训练配置及其哈希。训练只需指定数据集路径，省略的数据摘要由程序计算并冻结；显式摘要和续训绑定仍严格核验，旧封存运行仍能重跑和续训。

准备新采集数据统一使用 `fh5 collection-bc-prepare --config configs/collection-bc.example.json --output runs/numeric-candidate`。v2 配置合并来源、筛选规则和观测设置，一次生成 `selection.json`、训练/开发 `dataset.json` 与独立 `evaluation.json`；质量核验和尝试分组仍由用户决定。旧 `collection-dataset` 创建入口及独立的 v1 准备配置已退役，旧选择文件继续通过 `collection-dataset-review` 离线复核。后续 `temporal-train` 的训练参数和恢复方式不变，见[准备与迁移说明](docs/collection-datasets.md)。

CPU 数值 BC 已接入[训练状态恢复](docs/bc-resume.md)：资源停止时在完整更新边界保存网络、Adam 和随机数状态，在新目录继续原预算中的剩余更新；已完成训练的检查点可直接重新验证和发布。恢复优先使用 `learner/learner.json`，调度报告缺失或截断不阻止接续，可选日志和统计缺失也不丢弃已封存进度。连续 6 次与 2＋4 次 CPU 更新已验证精确一致；独立审阅和完整回归尚未完成，强制进程终止和 CUDA 续训未获验收。

T26 / #27 从实体 XInput 手柄读取原始动作，与 RGB 和遥测同步记录；双踏板、手刹、争用及失焦等片段保留并排除。`fh5 input-devices` 查看设备，`fh5 demonstrate` 被动采集，`fh5 demonstration-dataset` 按独立回合导出无参考/参考辅助视图及未来轨迹监督。校准、质量审阅和命令见[同步示范说明](docs/demonstrations.md)，真实短段及失败排除见[验收记录](docs/validation/t26-demonstrations.md)。这一切片不训练或执行驾驶策略。

## 开发检查

T05 / #6 提供[局部传统路线跟随](docs/tracking.md)：`fh5 track` 默认仅检查配置，显式 `--live` 才连接控制。连续弯道、减速和有界纠偏目前为合成环境验证；真实连续弯道验收仍待完成，不属于 BC/RL 成果。

T11 / #12 已提供[恢复监督回放](docs/recovery.md)：通过合成任务/UI 信号检查释放、倒带确认、历史隔离和限次停止，命令为 `fh5 recovery-replay`。它不操作游戏；自主倒带与恢复后实际驾驶尚未验收。

```powershell
uv run --locked pytest
uv run --locked mypy
uv run --locked ruff check .
uv run --locked ruff format --check .
```

测试通过实验运行入口验证录制、回放和故障行为。录制与回放只需 Python 标准库；实机控制使用可选 `control` 依赖和单独安装的驱动。`src/fh5/` 是代码及报告模板，`tests/` 是合成输入测试，`configs/` 是配置示例，`runs/` 是不纳入 Git 的本地实验结果。

## 当前文档

| 文档 | 用途 |
|---|---|
| [PRD](docs/PRD.md) | 当前目标、范围、功能与验收基线；实施入口 |
| [范围记录](docs/scope-decisions.md) | Q1–Q15 决策与用户补充要求 |
| [术语表](CONTEXT.md) | 统一领域词义 |
| [驾驶学习主方案](docs/driving-learning-design.md) | 基线、BC→SAC、自主学习循环与实施主线 |
| [奖励与有效性设计](docs/reward-and-validity-design.md) | 奖励投机防护、恢复分段、有效成绩及反例验证 |
| [倒带研究](docs/rewind-research.md) | 恢复能力的依据与待实测项 |

## 设计决定与研究

- [ADR 0001：首版使用已知道路信息与本车遥测](docs/adr/0001-known-route-first.md)
- [ADR 0002：将倒带恢复与正式驾驶分开](docs/adr/0002-separate-driving-and-recovery.md)
- [ADR 0003：成绩有效性先于性能排名](docs/adr/0003-validity-before-performance.md)
- [原始可行性评估](docs/feasibility-plan.zh-CN.md)、[参考项目审计](docs/reference-project-audit.md)、[FH5 接口研究](docs/fh5-environment-research.md)、[RL 方法研究](docs/rl-methods-research.md)

研究文档保留原始证据和历史候选；早期选车、实施顺序与排期建议以当前 PRD 和用户最新决定为准。

### 离线模仿学习

#11 已增加 [SAC 转移、预热与离线更新](docs/sac-learning.md)：在合成数值经验上核验完整 BC 冻结预热后，`sac-train` 实际更新策略、温度、编码器和双 Q，`sac-policy-replay` 重载冻结策略。自主游戏采样循环及实机收益仍待完成。

#13 已增加 [SAC 中断与续训](docs/sac-resume.md)：新快照封存数值经验、完整学习状态和训练历史，`sac-resume` 在新目录继续同一学习过程；`stop.request` 在完整更新边界保存。当前只支持 CPU 合成经验与兼容契约，实机重新入场仍待接入。

#11 已接通[有界合成采样与学习循环](docs/sac-cycle.md)：冻结策略控制响应动作的测试环境，独立结算后追加经验、续训并在下一次尝试换版；数值推理先于归档，失败与排除记录保留。它验证循环软件，不代表 FH5 驾驶或默认版本晋升。

新 BC 训练统一走下述数值 Δt 路径；旧 `bc-train` 已退役，旧模型仍可通过 `fh5 bc-replay` 离线诊断。历史数据可在满足时间契约后导入并重新训练，不转换旧权重或改写原件，见[迁移说明](docs/bc.md)。首轮旧模型结果保留在 [T27 验证](docs/validation/t27-bc.md)。

#35 新增 `fh5 temporal-prepare`、`temporal-train`、`temporal-replay`：历史画面仅在导入时解码，训练与推理使用数值像素和明确的帧间 Δt。支持实际/固定时间对照及冻结输入/预测核验，见[数值 Δt BC](docs/temporal-bc.md)和[离线验证](docs/validation/t33-temporal-bc.md)。数值 Δt BC 已接入实时驾驶软件路径；首轮结果未证明 Δt 收益或起步能力，实机驾驶仍待验收。

#34 已有 [DXGI 数值采集基础入口](docs/dxgi-capture.md)，`fh5 capture-dxgi` 默认只校验配置。独立采集/预处理、最新待处理槽和带 QPC 时间的历史支持软件验证；原生动态采集、性能对照与页面交互仍待验收，不构成实时驾驶通过。

#36 已有[容错数值决策软件切片](docs/realtime-decisions.md)：常驻推理、缺帧跳过、绝对动作租期和独立监督通过故障回放及真实线程测试。`fh5 realtime-shadow` 默认只校验，显式 `--live` 才组合 DXGI、UDP、独立任务几何和冻结 Δt 模型做只读预测；不会连接虚拟手柄。组合通过合成像素与回环 UDP 验证，实际 FH5 的 10/20 Hz 性能和页面交互仍待验收。

当前模型驾驶入口为 `fh5 realtime-drive`，使用数值 Δt BC。只读预测与驾驶共用 `configs/realtime-drive.example.json`：先保持 `shadow: null` 运行 `realtime-shadow`，取得对应实测记录后只补入证据引用，再运行 `realtime-drive`。两者默认只校验；影子的 `--live` 只采集和预测，驾驶的 `--live` 才可能连接控制，且仍需通过资格核验。用法与旧影子配置迁移见[数值驾驶说明](docs/realtime-decisions.md#有界驾驶命令与条件绑定)。旧 `policy` 在线入口已退役，旧模型和 JSON 不自动转发；旧录制仍可离线回放，见[迁移说明](docs/policy-driving.md)。
