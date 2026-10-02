# 持续采集数据快照（T36 / #38）

状态：已实现封存来源的固定选择、核验记录绑定、关联尝试分组、标签筛选、覆盖报告与数值 BC 导出，接通 Δt BC 训练、冻结比较、独立最终评估及采集优先的训练调度。**真实共享负载和新数据验收仍待完成；#38 不关闭。** 本轮仅使用合成源，没有开启 FH5/Steam 或真实采集设备。

## 操作入口

```powershell
uv run --locked fh5 collection-bc-prepare --config configs/collection-bc.example.json --output runs/numeric-candidate
uv run --locked fh5 collection-dataset-review runs/numeric-candidate/selection.json --report runs/numeric-candidate/rechecked.html
```

`collection-bc-prepare` 是唯一创建入口。使用 [`configs/collection-bc.example.json`](../configs/collection-bc.example.json) 的 v2 配置，在同一层填写 `seed`、`sources`、`rules` 与动作历史、航点布局和可选 `reference`。命令直接从已核验来源完成选择和数值导出，不再先创建独立 selection 目录，也无需向准备配置手抄 selection 路径或 SHA-256。

配置中的路径相对配置文件。先替换来源与核验文件路径；核验格式见 [`configs/collection-review.example.json`](../configs/collection-review.example.json)。核验中的 `session_sha256` 仍绑定对应 `recording/session.json` 原始字节。质量、尝试分组和集合划分由用户复核，准备命令不会代为确认。每次导出使用新目录，追加核验或修改筛选规则生成新快照。该接口只处理封存文件，不抢占游戏、UDP 或输入设备。

输出中的 `selection.json` 保留原选择快照格式；`dataset.json` 和 `evaluation.json` 分别供训练/开发与最终评估使用。selection 固定当时 `index.json` 已发布的块引用及哈希，绑定 session 中的软件来源、配置、输入档案和像素/时间契约。离线复核只读取这组来源；后来追加的块和修改后的外部 review 不改变已有快照。复核仍依赖原录制目录中的源块与数值帧，不能移动或删除；缺失/损坏依赖会失败，selection 不是全量备份。哈希用于完整性检查，不是阻止所有文件一起被改写的签名。

旧 `collection-dataset` 创建命令已退役，调用时返回迁移错误。旧 `collection-bc-prepare` v1 配置中的 `dataset` / `dataset_sha256` 不再接受：将原选择配置的 `seed`、`sources`、`rules` 与原观测设置合并为 v2，并按新配置位置调整相对路径。旧选择快照无需改写，仍可用 `collection-dataset-review <旧 dataset.json> --report <新报告.html>` 离线复核；该命令也读取新输出的 `selection.json`。已有数值数据集、训练配置、模型和恢复契约不变。

## 核验与尝试分组

核验记录给出按轮询 `sequence` 表示的半开范围 `[start_sequence, end_sequence)`。明确 attempt ID、关联 group 及 train/development/evaluation。无独立证据时，同一 session 的全部尝试使用同一个 group；重开/倒带的强关联尝试填写 `related_attempts`，系统拒绝把这些尝试或同一 group 分到不同集合。将单个 session 拆成独立 group 必须有 `independence_evidence`。这些声明需要实际复核，程序不会从游戏时钟跳变自动推断已经独立完赛。

当前来源使用同一主机的单调时钟；不同录制中的已核验尝试时间窗重叠时拒绝混合，不能仅换一个 session ID 就声称独立。跨系统重启后时钟域不确定的数据需要另行核验/扩展契约，不能直接假定时间可比较。

每个尝试可细分 trusted/failed/unknown 区间，保留证据与原因；没有覆盖的区间视为未知。`conditions_verified` 表示已经核对采集条件、导航及输入归属/采纳依据，不以文件存在代替实机证据。未知条件不能产生合格 BC 标签。道路类型可标 straight/left_curve/right_curve/unknown；方向输入事件不能冒充已核实弯道类型。

## 标签与用途

观测时刻取当次采集检查时刻。监督标签采用**紧邻下一次轮询**中、发生在观测之后且处于声明延迟上限内的实际实体输入；保存标签轮询/可用时刻、块和序号。最后一行或下一行缺失、跨尝试/区间/前向片段时不配标签，不能用最近过去动作偷换。当前标签不会作为该观测的历史输入；数值导出为 actor 单独构造截止之前已知的动作历史。

原始字节重新解码并核对保存的遥测；原始 XInput 重新映射并核对动作。损坏存储依赖使导出失败；真实源记录中已有的坏包可以隔离，保留前后正常部分。图像历史必须满足数值契约及原片段边界；核验区间改变或归档序号缺失后重新建立历史。

可信区间中且处于配置的速度/动作范围内的样本标为 BC 可用。失败与超范围样本保留实际标签及排除原因，**不裁剪动作再配原状态**。动力学与 Q 资格暂不开放：下一步还需构造并验证动作—响应对齐，Q 另需可靠奖励和终止。原始记录始终保留，不因筛选被删除。

每次尝试使用固定种子的 reservoir 抽样，示例配置每次尝试选择最多 1000 个同步样本；用户需按本次实验决定抽样预算。保留入选数量、候选数与省略数，样本数量不是数据足够或模型可驾驶的保证。

本次只合并准备流程，数据读取和导出中的已有固定门槛尚未全部清理，资源要求以[资源策略](resource-policy.md)为准。复核报告必须使用新的 `.html` 路径及同名 `.json`，不得覆盖已有文件或写入来源录制目录；例如复核 `selection.json` 时不要使用 `selection.html`，因为其 JSON 摘要会与输入重名。

## 覆盖与就绪报告

准备报告的 `selection` 部分直接提供覆盖与就绪信息，无需再运行复核才能查看；`collection-dataset-review` 也可独立重建这份选择报告。报告包含每次尝试的连续左右转向、油门/刹车/滑行、速度区间及已核验弯道事件。起步定义为低于 1 km/h 后越过该阈值且伴随油门；松 RT 定义为正纵向输入转为零或负。中断/无效区间切断事件，相邻帧不会各算一次转弯。阈值是版本化的软件统计定义，不能替代赛程能力评估。

补录建议只依据 train/development；evaluation 的行为覆盖不出现在就绪报告中，也不参与补录建议。报告仅展示最终留出组数和样本数；当前数据准备不会使用它调参。`ready_for_software_training` 只说明选中了合格训练和开发样本，`real_candidate_ready` 仍为 false。合成源或未冻结来源标 `diagnostic_only`，不能作为新采集的人工驾驶示范。

## 导出数值历史并训练

同一次 `collection-bc-prepare` 创建冻结选择并导出数值历史、本车状态、严格因果动作历史及配对参考视图。每帧逐字节复制 RGB 数值数据并按内容去重，不经过 JPEG，也不再次缩放；保留原像素尺寸、时间和不同来源的帧身份。准备命令不执行训练，也不改动原录制。

动作历史取每个偏移截止之前已获得且发生在当前核验区间内的实际输入；没有合格输入时保留缺失 mask。监督标签仍为决策之后的下一轮询，与 actor 的历史分离。本车世界坐标、来源身份、尝试 ID、标签时刻与筛选原因不进入 actor。

默认参考缺失，两种视图的参考 mask 均为空。可选 `reference` 为 `{"route_file":"../runs/independent-route/route.json","independence_evidence":["说明为何独立于当前训练/留出尝试"]}`；先绑定资产哈希，再加载复制后的路线包。禁止与当前采集 session 同源；其他独立性依赖给出的证据，不能仅凭哈希不同就认定独立。参考只生成局部航点，其与无参考视图保持同组，不能把未核实路线当作合法进度依据。

输出 `dataset.json` 只含 train/development，`evaluation.json` 单独保留最终留出。准备报告不显示最终行为或误差。后续训练仍使用独立配置：按 [`configs/temporal-bc.example.json`](../configs/temporal-bc.example.json) 填写，`dataset` 指向新 `dataset.json`，`dataset_sha256` 取准备报告的 `snapshot_sha256["dataset.json"]`，并明确训练种子、更新步数、批量、学习率和设备；`time_mode` 使用 `actual`，固定时间对照按[数值 Δt BC 说明](temporal-bc.md)设置。然后运行 `temporal-train --config <配置> --output <新模型目录>` 和 `temporal-replay --model <模型目录> --dataset <数值数据集> --report <新回放.html>`。模型读取内存中的 RGB 数值，PNG 只用于离线报告预览。参考缺失不声称已经验证有参考驾驶；合成来源始终带有 `diagnostic_only`。

## 比较冻结候选与最终留出

```powershell
uv run --locked fh5 collection-bc-assess --config configs/collection-assessment.example.json --output runs/candidate-development
```

配置绑定数据文件和候选 `model.json` 原始字节的 SHA-256。可选 `baseline` 使用与 `candidate` 相同的目录/哈希格式；缺省时仍提供复制最近动作的诊断。所有输出必须是新目录，且不能位于任何模型目录或输入数据目录内。输出含逐观测 RGB 预览、两种模型预测、实际示范动作、此前动作、时间及监督排除原因；预览不进入模型输入。

`development` 使用 `dataset.json`，只预测和统计开发组，不评估训练组或最终留出。选定并冻结候选后，以新配置设置 `mode: "final"`，数据改为 `evaluation.json` 及其哈希。数值导出已把最终文件哈希写入训练快照，后续训练继承该绑定；不能训练完成后重新指定一个留出集合。旧导出缺少绑定时应重新生成新版本及候选，不能改写旧模型补造来源。

最终评估拒绝候选训练/开发组与留出组身份、同源序号区间或同一主机单调时钟窗口重叠。缺少时间范围的旧基线不能证明独立性。比较还检查像素、时间历史、参考路线、车况、输入映射与动作/速度包络等契约；旧来源或受污染基线标为不可比，不给它计算分数，也不重新缩放以强行比较。开发比较允许基线以前使用同一开发集，但该数据不能曾是其训练集。

分别报告有参考/无参考视图及起步、左右转向、松 RT、滑行、制动的双轴误差与独立组数。此处分层数量是观测数，连续事件覆盖仍看数据快照报告。缺失行为的计数为零、误差为 `null`；不能解释为零误差或已经学会。候选和可比基线均重新加载、复算，容差为 `1e-6`，原模型文件保持不变。

`final` 报告标记 `selection_allowed: false`，没有训练更新或自动晋升。最终结果一旦被用于调参或补录选择，下一次独立结论必须使用新的未见留出；工具不提供跨项目的全局访问次数登记。离线动作误差不证明真实驾驶改善，合成模型继续保持 `diagnostic_only`。

采集期间使用 [`collection-bc-train` 调度入口](learning-schedule.md)：CPU 小任务按采集健康和预算并行，CUDA 先采用交错基线，压力下暂停或有界退出。当前已经能在后台独立合成采集继续封存时，用固定快照完成 CPU 训练与加载回放；该原型不代表 4K 游戏负载通过。实际 4K 条件、共享资源预算与新人工多次驾驶仍按 #34/#37 验收。
