# FH5 歌利亚自主驾驶与学习实验

目标：让固定调校的兰博基尼第六元素特别版 X 999 自主完成歌利亚，再通过强化学习改善速度、稳定性和竞速表现。使用官方路线与检查点；当前空场蓝图为 **105657219**，干燥季节、晴朗、固定早晨。

当前主线是 **DXGI 数值画面与遥测/输入 → BC 初始化 → 冻结检查与短段驾驶 → SAC 学习 → 独立评估和版本保存**。优先 4K 游戏来源，默认缩放为 480×270 RGB；模型接收数值像素和实际帧间 Δt。

已取得真实遥测、低速控制与停止释放证据，并实现独立采集、数值 BC、SAC 更新/恢复及冻结评估的软件链路。`LearningLoop` 的版本 1–3 使用合成环境，版本 4 已接入原生短段适配器；**新 4K 多圈数据、游戏共享负载、实机自主学习闭环和驾驶改善仍待验收，尚未证明自主完成歌利亚。**

## 环境与入口

Windows、Python 3.12、uv；从仓库根目录使用 PowerShell：

```powershell
uv sync --locked --extra capture --extra learning
uv run --locked fh5 --help
```

实体手柄采集按[示范与输入校准](docs/guides/capture/demonstrations.md#采集与校准)准备档案；模型实际控制另需 `control` extra 和 ViGEmBus，见[控制安装](docs/guides/runtime/control.md)。实验使用新的输出目录，`runs/` 不纳入 Git，已有录制、模型和依赖目录保留。

| 顺序 | 当前操作入口 | 完成后进入下一步的依据 |
|---|---|---|
| 1. 采集 | [独立持续采集](docs/guides/capture/continuous-collection.md)：`collection-prepare/start/status/stop/review` | 已封存录制、真实采集条件和输入核验 |
| 2. BC | [准备数据](docs/guides/bc/collection-datasets.md)：`collection-bc-prepare`；[训练与恢复](docs/guides/bc/training.md#temporal-bc)：`temporal-train/replay` | 按独立组划分数据，冻结候选并核对训练/运行契约 |
| 3. 运行 | [只读检查与有界驾驶](docs/guides/runtime/realtime-decisions.md)：`realtime-shadow/drive` | 匹配的只读证据、独立局部任务和驾驶资格 |
| 4. SAC | [初始化与更新](docs/guides/sac/sac-learning.md)，再到[连续学习循环](docs/guides/sac/learning-loop.md) | 合格经验、完整 learner、释放资源后的阶段交接 |
| 5. 评估 | [冻结评估](docs/guides/evaluation/evaluation.md)与[候选保存](docs/guides/evaluation/candidates.md#candidate-store) | 全部尝试、独立有效性依据和事先冻结的比较规则 |

命令参数、配置示例、停止/恢复方式与剩余验收见对应指南。`realtime-shadow` 和 `realtime-drive` 默认只检查配置；显式 `--live` 才连接设备，驾驶另须通过资格检查。原生学习同样须显式 `live=True`，目前限定短段；自动倒带和全程能力另行验收。

[文档索引](docs/README.md)汇集当前指南、设计、研究和历史资料。产品与验收基线见 [PRD](docs/design/PRD.md)，术语见 [CONTEXT.md](CONTEXT.md)，当前任务见 [GitHub Issues](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues)。旧模型及录制的读取方式见[兼容与迁移](docs/archive/legacy-models.md#bc)，原始资料不自动转换或覆盖。

## 开发

`src/fh5/` 按遥测、观测、采集、驾驶、学习、评估与报告组织；`tests/` 对应这些职责，`configs/` 保持配置路径稳定。依赖方向与兼容入口见 [代码架构](docs/architecture.md)，贡献约定见 [AGENTS.md](AGENTS.md)。

```powershell
uv run --locked pytest
uv run --locked mypy
uv run --locked ruff check .
uv run --locked ruff format --check .
```

测试以实验运行入口、真实临时文件和小型模型验证软件行为；原生设备和驾驶能力另记实机证据。
