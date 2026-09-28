# FH5 歌利亚自主驾驶与学习实验

使用用户固定调校的兰博基尼第六元素，从遥测与已知路线起步，建立能自行采样、训练、评估并改进驾驶策略的实验系统。首版聚焦无对抗歌利亚，后续探索对手竞速和路线泛化。

当前实现：T01 遥测录制与离线回放已获得真实驾驶记录；T02 已在用户第六元素 X 999 上完成低速起步、左右转向、刹停及 F8/失焦/暂停释放验证。峰值约 9.58 km/h，尚未实现路线跟随或自主驾驶。见 [T02 控制说明](docs/control.md) 与 [实机验证记录](docs/validation/t02-control.md)。七天是首轮探索窗口，按可运行里程碑推进。

## 安装与运行

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

## 开发检查

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
