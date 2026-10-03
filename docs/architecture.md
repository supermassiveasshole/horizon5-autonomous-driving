# 代码结构与依赖

`fh5.experiment.run_experiment` 是实验运行与行为测试的统一入口。CLI 在 `cli.py` 和 `commands/` 装配具体环境；参数定义集中在 `commands/parser.py`，按工作流分组。包的 `__init__.py` 不加载设备或整套工作流。

| 位置 | 负责什么 |
|---|---|
| `telemetry/` | FH5 包类型、解码、录制、重放与断流/跳变诊断 |
| `observation/` | 数值图像契约、因果动作历史、多模态观测、路线与道路研究 |
| `capture/` | DXGI 与图像管线、采样时刻；旧彩色录制用于回放兼容 |
| `collection/` | 独立采集进程、封存、示范筛选与 BC 数据准备 |
| `driving/` | 标定、赛事操作、恢复与传统跟随；`realtime/` 负责数值策略执行和时效监督 |
| `learning/` | `bc/` 模仿初始化、`sac/` 强化学习、`loop/` 采样/更新/评估编排；共享训练资源与存储能力放本层 |
| `evaluation/` | 完整尝试、奖励审计、冻结评估和候选版本管理 |
| `artifacts/` | 文件完整性、持久化、流式文档读取与证据用途 |
| `reporting/` | 遥测、数值、实时报告和可选诊断展示 |

## 修改应落在哪里

- 纯遥测实现仅依赖标准库、`telemetry.packet` 与 `result.RunResult`。实验入口在得到基础遥测结果后组合控制、路线、视觉证据并生成报告；数据类型不反向导入实验分发器。
- 数值图像、时间戳与历史有效性属于 `observation.numeric`；录制/推理属于 `observation.recording`。增加一种输入信息先修改其契约和消费者，而不是给类型模块增加工作流分派。
- 通用文件能力由 `artifacts.io` 维护。训练和评估不从采集工作线程获取 JSON 编码、原子发布或文件写入函数。
- 采集运行循环负责采样与封存；进程启动、状态和停止请求统一由 `collection.process` 负责。冻结采集包复制完整源码及模板，从自己的安装环境运行。
- 外部 adapter 与所属能力放在一起，如 `capture.windows`、`driving.windows` 和 `learning.loop.native`。Windows、Torch 和截屏依赖继续按实际使用延迟加载。
- 编排工作流可以调用其他工作流；底层数据与文件模块不依赖这些编排。现有部分工作流仍借助 `run_experiment` 组合实验，这不代表包依赖已完全无环。

## 兼容与验证

保留 `fh5.experiment` 的 `Packet`、`Record`、`Replay`、`RunResult`、`run_experiment`，以及 `fh5.cli:main` 和已有命令。其他内部 Python 路径已按职责迁移，当前示例使用新路径；录制、像素、模型及检查点格式未因此升级。

`tests/` 按同样职责组织，独立测试辅助程序在 `tests/support/`，其余 fixture 贴近对应行为测试。功能测试仍通过实验入口和 CLI；迁包还需核对后台进程模块名、相邻 HTML 模板、源码指纹覆盖与安装后的入口。软件测试不替代 FH5 驾驶验收。

文档从 [索引](README.md) 进入：`guides/` 写当前操作，`design/` 写约定，`research/` 保留来源调查，`archive/` 保留历史范围及旧资产迁移，`validation/` 保留实际证据。新实现修订所属指南；避免为同一能力不断增加平行说明。
