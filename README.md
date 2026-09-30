# FH5 歌利亚自主驾驶与学习实验

使用用户固定调校的兰博基尼第六元素，从遥测与已知路线起步，建立能自行采样、训练、评估并改进驾驶策略的实验系统。首版聚焦无对抗歌利亚，后续探索对手竞速和路线泛化。

当前实现：T01 遥测录制与离线回放已获得真实驾驶记录；T02 已在用户第六元素 X 999 上完成低速起步、左右转向、刹停及 F8/失焦/暂停释放验证。峰值约 9.58 km/h，尚未实现路线跟随或自主驾驶。见 [T02 控制说明](docs/control.md) 与 [实机验证记录](docs/validation/t02-control.md)。七天是首轮探索窗口，按可运行里程碑推进。

T03 开发中：[赛事生命周期](docs/events.md) 已接入 Windows 菜单适配器和 `fh5 event`。空场蓝图 `458 574 769` 的程序探测完成三次静止起跑核验及两次连续重开，约 43 秒，全程零驾驶输入；固定环境与持续采样尚未验收，见 [T03 实机记录](docs/validation/t03-event.md)。人工整圈也已保存，但因使用倒带不计入有效基线。`configs/goliath-458574769.json` 是录制快照；菜单探测另需本地页面模板及配置。

当前实验条件更新为蓝图 `105 657 219`：干燥季节、晴朗、固定早晨，配置为 `configs/goliath-fixed-v2.json`。已完成一次赛途中暂停后自动重开，从准备到起点核验约 12.91 秒；旧蓝图记录保留原条件。持续无人值守验收仍未完成。

T04 局部数据验收完成：[局部路线工具](docs/routes.md) 可从连续记录导出参考轨迹，独立保存走廊与检查点依据，并回放连续定位与已确认进度。已保留约 801 米历史参考，拱门后约 19.55 米的保守内部区域 v2 已经独立实测通过：653 个连续样本确认全段，窗口峰值约 9.97 km/h。`fh5 route-check` 保留完整记录、失败与版本依据；受限速度采集不等于学习策略驾驶，局部尝试有效性现由 #7 实现，见 [T04 数据记录](docs/validation/t04-route.md)。下一步按已批准的[多模态学习规格](docs/multimodal-learning-spec.md)推进：含道路/导航的真实截图历史、本车状态、因果动作历史及可选历史航点进入 BC→SAC；先做可信同步示范和离线 BC，训练与评估实际覆盖无参考条件，见[视觉导航增量](docs/visual-navigation-spec.md)。当前 #26 的 v1 保留必选参考回放，#31 已实现可选/禁用参考的 v2 观测，已完成 #28 离线 BC 初始化但未通过实机驾驶验收。精确分割/米制道路保留为可选研究，实机控制和成绩有效性另行验收。

T06 / #7 已加入 `fh5 attempt-review`：完整尝试、前向片段和恢复排除区分别留存，独立证据决定局部结论，正式全程仍单独待核验。支持已有记录离线审核，不发送控制；使用方式见 [局部尝试说明](docs/attempts.md)，真实覆盖与限制见 [T06 验证](docs/validation/t06-attempts.md)。

## 安装与运行

T07 / #8 新增[物理时间奖励与终止结算](docs/rewards.md)：`fh5 reward-replay` 重算历史局部片段，`fh5 reward-audit` 生成完整合成反例与回报排序。独立有效性、奖励标签和正式成绩保持各自结论；未启动 SAC 或新的实机驾驶。

T31 / #33 已加入独立的[数值图像输入与精确回放](docs/numeric-images.md)：旧图像仅在离线准备时解码，冻结策略接收数值 RGB，数值存档在后台执行。真实历史数据已有 200 个观测精确回放；页面交互仍待验收，旧实机 `policy` 尚未迁移。参见 [T31 验证](docs/validation/t31-numeric-images.md)。

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

#37 新增[持续被动采集](docs/continuous-collection.md)：冻结代码/依赖后独立后台运行，支持状态、停止与已封存数据恢复。原生 DXGI/UDP/XInput 组合使用数值像素；后台生命周期已用合成输入验证，真实多次驾驶与 4K 性能仍待验收。

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

安装 `learning` 可选依赖后，通过 `fh5 bc-train --config configs/bc.example.json --output runs/bc-first` 训练固定预算的多模态 BC；`fh5 bc-replay` 重放冻结模型。两者不发送游戏输入。数据、参考遮蔽、模型与误差解释见 [BC 说明](docs/bc.md)，首轮结果见 [T27 验证](docs/validation/t27-bc.md)。

#35 新增 `fh5 temporal-prepare`、`temporal-train`、`temporal-replay`：历史画面仅在导入时解码，训练与推理使用数值像素和明确的帧间 Δt。支持实际/固定时间对照及冻结输入/预测核验，见[数值 Δt BC](docs/temporal-bc.md)和[离线验证](docs/validation/t33-temporal-bc.md)。首轮结果未证明 Δt 收益或起步能力，尚未接入实时驾驶。

#34 已有 [DXGI 数值采集基础入口](docs/dxgi-capture.md)，`fh5 capture-dxgi` 默认只校验配置。独立采集/预处理、最新待处理槽和带 QPC 时间的历史支持软件验证；原生动态采集、性能对照与页面交互仍待验收，不构成实时驾驶通过。

#36 已有[容错数值决策软件切片](docs/realtime-decisions.md)：常驻推理、缺帧跳过、绝对动作租期和独立监督通过故障回放及真实线程测试。`fh5 realtime-shadow` 默认只校验，显式 `--live` 才组合 DXGI、UDP、独立任务几何和冻结 Δt 模型做只读预测；不会连接虚拟手柄。组合通过合成像素与回环 UDP 验证，实际 FH5 的 10/20 Hz 性能和页面交互仍待验收。
