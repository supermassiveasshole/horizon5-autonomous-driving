# FH5 歌利亚自主驾驶与学习实验

目标：让固定调校的兰博基尼第六元素特别版 X 999 自主完成歌利亚，再通过强化学习改善速度、稳定性和竞速表现。使用官方路线与检查点；当前空场蓝图为 **105657219**，干燥季节、晴朗、固定早晨。

当前主线是 **DXGI 数值画面与遥测/输入 → BC 初始化 → 冻结检查与短段驾驶 → SAC 学习**。优先 4K 游戏来源，默认缩放为 480×270 RGB；模型直接接收数值像素和实际帧间 Δt，采集、预处理、推理、存档各自运行。

## 已做到哪里

| 环节 | 已有证据 | 尚未完成 |
|---|---|---|
| 游戏接入 | 真实遥测、手柄轴/踏板校准、低速起停与停止释放、局部路线记录 | 新管线实际模型驾驶 |
| 数值采集与 BC | 独立后台采集、封存数据准备、真实 CPU 模型更新/加载/续训的软件验证 | 新 4K 多圈数据、实际共享负载与驾驶收益 |
| BC→SAC | 真实 SAC 更新、完整 checkpoint 续训、响应动作的合成环境采样与冻结评估 | FH5 原生自主学习闭环、自动恢复后的驾驶与有效完赛 |

本项目尚未证明学习策略能自主完成歌利亚。各环节的完整验收见对应文档和 [GitHub Issues](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues)。

## 环境

Windows、Python 3.12、uv；从仓库根目录使用 PowerShell。基础录制/回放使用标准库，训练与原生数值采集按需安装额外依赖：

```powershell
uv sync --locked --extra capture --extra learning
uv run --locked fh5 --help
```

实体手柄采集不需要虚拟手柄驱动。模型实际控制另需 `control` extra 和已安装的 ViGEmBus，按[控制说明](docs/control.md)配置。首次手柄校准保留 `input-devices` / `demonstrate`，步骤与所需 `events` extra 见[输入校准](docs/demonstrations.md#采集与校准)。

新实验使用新的输出目录。`runs/` 不纳入 Git；已有数据、模型和依赖目录应保留。下列命令是操作顺序，需先替换本机路径并完成各步注明的核验。

## 1. 独立采集人工驾驶

在 FH5「设置 → HUD 与游戏」开启 Data Out，目标 `127.0.0.1:5300`。固定原车、调校、辅助和追尾远视角；在 `configs/capture-dxgi.example.json` 核对分辨率、HUD 等真实条件，未知项保持未核验。准备已校准的实体手柄档案 `runs/input-profile.json`。

```powershell
uv run --locked fh5 collection-prepare --capture-config configs/capture-dxgi.example.json --input-profile runs/input-profile.json --output runs/collector
uv run --locked fh5 collection-start runs/collector --output runs/collection-001 --live
uv run --locked fh5 collection-status runs/collection-001
uv run --locked fh5 collection-stop runs/collection-001
uv run --locked fh5 collection-review runs/collection-001/recording --report runs/collection-001-reviewed.html
```

准备一次冻结安装，之后可更换 `--output` 重复采集。采集进程独立于聊天和开发源码运行；正常驾驶可包含多次起步、重开与跑圈，程序保留片段边界。查看状态确认停止后，再审阅已封存记录。软件实现与实机待验收项见[持续采集](docs/continuous-collection.md)。

## 2. 准备数据并训练 BC

参照 `configs/collection-review.example.json` 核对各次尝试和质量区间，按独立组划分 train/development/evaluation。示例不会自动确认正常驾驶；倒带、失败和未知片段必须如实保留。将来源与核验文件填入 `configs/collection-bc.example.json`。

```powershell
uv run --locked fh5 collection-bc-prepare --config configs/collection-bc.example.json --output runs/numeric-candidate
uv run --locked fh5 temporal-train --config configs/temporal-bc.example.json --output runs/temporal-model
uv run --locked fh5 temporal-replay --model runs/temporal-model --dataset runs/numeric-candidate/dataset.json --report runs/temporal-reloaded.html
```

训练配置的 `dataset` 指向准备结果；路径相对配置文件，数据摘要自动绑定。`temporal-replay` 使用训练时同一数据集复核冻结结果；独立最终留出 `evaluation.json` 使用 `collection-bc-assess`，不能混用。示例步数、批量和抽样数是待选择的实验参数，不代表数据或训练已足够。

准备器逐条读取图像、按内容去重保存，不再用累计帧字节量拒绝导出；样本数与元数据大小等历史限制仍待清理，已知边界见[数据准备](docs/collection-datasets.md)与[资源策略](docs/resource-policy.md)。需要边采集边训练时，用[采集优先调度](docs/learning-schedule.md)的单份 `collection-learning.example.json`；CPU 停止后的接续见[BC 恢复](docs/bc-resume.md)。

## 3. 冻结候选进入只读检查与驾驶

修改 `configs/realtime-drive.example.json` 中的模型目录、设备和已核验局部路线；采集条件须与训练一致，初始保留 `shadow: null`。先运行只读配置检查：

```powershell
uv run --locked fh5 realtime-shadow --config configs/realtime-drive.example.json --output runs/shadow-001
uv run --locked fh5 realtime-drive --config configs/realtime-drive.example.json --output runs/drive-001
```

这两条命令默认都不开设备。准备好游戏中的局部起点后，影子命令加 `--live` 才采集与预测，仍不发送控制。取得匹配的实测只读记录后填写 `shadow.directory`，查看驾驶检查的 `qualification` 和失败原因；哈希自动绑定，显式旧哈希仍严格核验。符合条件后，驾驶命令加 `--live` 才可能接管，受现有短段、速度、动作时效和停止保护约束。完整步骤见[数值驾驶](docs/realtime-decisions.md)。

候选来源、动作历史和图像契约必须匹配。配置验证成功、离线误差下降或模拟执行器收到动作，都不等于 FH5 已实际起步或驾驶通过。

## 4. SAC 与自主迭代

BC 负责初始化，自主采样和强化学习是后续主线。当前可运行真实 CPU SAC 更新、停止后继续、合成环境中的采样—学习—冻结评估；**现有 `LearningLoop` 仅支持合成外部环境**，不能直接当作 FH5 自动跑圈程序。

- [SAC 初始化与更新](docs/sac-learning.md)：合法转移准备、critic 预热、`sac-train` 与冻结回放。
- [完整学习状态恢复](docs/sac-resume.md)：`sac-resume` 继续网络、优化器、目标网络、温度和随机数状态。
- [采样循环](docs/sac-cycle.md)与[连续学习循环](docs/learning-loop.md)：封存经验、更新预算、重复评估与恢复。
- [独立评估](docs/evaluation.md)、[候选比较](docs/candidate-selection.md)与[版本保存](docs/candidate-store.md)：可靠默认与探索候选分开，奖励不能自证有效成绩。

自动倒带、重开和 FH5 原生学习闭环仍需相应实机验证，不能用合成结果追认。

## 历史资料与研究入口

已有编码图像记录仍可[观测回放](docs/observations.md)、[导航回放](docs/navigation-observations.md)或[离线导入数值训练](docs/temporal-bc.md)。旧 `policy`、`bc-train`、`numeric-prepare`、`collection-dataset` 创建入口已退役，迁移分别见[旧驾驶](docs/policy-driving.md)、[旧 BC](docs/bc.md)、[数值包](docs/numeric-images.md)、[数据准备](docs/collection-datasets.md)。原始资料不自动转换或覆盖。

[遥测录制](docs/recording.md)、[赛事菜单](docs/events.md)、[路线工具](docs/routes.md)、[传统跟随](docs/tracking.md)、[像素道路研究](docs/perception.md)和[恢复回放](docs/recovery.md)保留独立用途。详细产品依据为 [PRD](docs/PRD.md)、[驾驶学习方案](docs/driving-learning-design.md)、[奖励与有效性](docs/reward-and-validity-design.md)；术语见 [CONTEXT.md](CONTEXT.md)，决策历史见 [范围记录](docs/scope-decisions.md)和 [ADR](docs/adr/)。

## 开发

`src/fh5/` 为实现与报告模板，`tests/` 为行为测试，`configs/` 为配置示例。贡献约定见 [AGENTS.md](AGENTS.md)。

```powershell
uv run --locked pytest
uv run --locked mypy
uv run --locked ruff check .
uv run --locked ruff format --check .
```

测试以实验运行入口、真实临时文件和小型模型验证软件行为；原生设备和驾驶能力另记实机证据。
