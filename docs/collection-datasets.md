# 持续采集数据快照（T36 / #38）

状态：已实现封存来源的固定选择、核验记录绑定、关联尝试分组、标签筛选及覆盖报告。**后续 Δt BC 导出、训练比较及真实新数据验收仍在实施；#38 不关闭。** 本轮仅使用合成源，没有开启 FH5/Steam 或真实采集设备。

## 操作入口

```powershell
uv run --locked fh5 collection-dataset --config configs/collection-dataset.example.json --output runs/dataset-001
uv run --locked fh5 collection-dataset-review runs/dataset-001/dataset.json --report runs/dataset-001/rechecked.html
```

配置中的路径相对配置文件。先替换示例路径；核验格式见 `configs/collection-review.example.json`。`session_sha256` 绑定对应 `recording/session.json` 原始字节。每次导出使用新目录，追加核验或修改筛选规则生成新快照。该接口只处理封存文件，不抢占游戏、UDP 或输入设备。

快照固定当时 `index.json` 已发布的块引用及哈希，绑定 session 中的采集软件、依赖、配置、输入档案和像素/时间契约。复核只读取这组来源；后来追加的块和修改后的外部 review 不改变已有快照。源块与数值帧仍由原录制目录持有，不能移动或删除；缺失/损坏依赖会失败，快照不是全量备份。哈希用于完整性检查，不是阻止所有文件一起被改写的签名。

## 核验与尝试分组

核验记录给出按轮询 `sequence` 表示的半开范围 `[start_sequence, end_sequence)`。明确 attempt ID、关联 group 及 train/development/evaluation。无独立证据时，同一 session 的全部尝试使用同一个 group；重开/倒带的强关联尝试填写 `related_attempts`，系统拒绝把这些尝试或同一 group 分到不同集合。将单个 session 拆成独立 group 必须有 `independence_evidence`。这些声明需要实际复核，程序不会从游戏时钟跳变自动推断已经独立完赛。

当前来源使用同一主机的单调时钟；不同录制中的已核验尝试时间窗重叠时拒绝混合，不能仅换一个 session ID 就声称独立。跨系统重启后时钟域不确定的数据需要另行核验/扩展契约，不能直接假定时间可比较。

每个尝试可细分 trusted/failed/unknown 区间，保留证据与原因；没有覆盖的区间视为未知。`conditions_verified` 表示已经核对采集条件、导航及输入归属/采纳依据，不以文件存在代替实机证据。未知条件不能产生合格 BC 标签。道路类型可标 straight/left_curve/right_curve/unknown；方向输入事件不能冒充已核实弯道类型。

## 标签与用途

观测时刻取当次采集检查时刻。监督标签采用**紧邻下一次轮询**中、发生在观测之后且处于声明延迟上限内的实际实体输入；保存标签轮询/可用时刻、块和序号。最后一行或下一行缺失、跨尝试/区间/前向片段时不配标签，不能用最近过去动作偷换。当前标签不会作为该观测的历史输入；导出因果动作历史属于下一实现步骤。

原始字节重新解码并核对保存的遥测；原始 XInput 重新映射并核对动作。损坏存储依赖使导出失败；真实源记录中已有的坏包可以隔离，保留前后正常部分。图像历史必须满足数值契约及原片段边界；核验区间改变或归档序号缺失后重新建立历史。

可信区间中且处于配置的速度/动作范围内的样本标为 BC 可用。失败与超范围样本保留实际标签及排除原因，**不裁剪动作再配原状态**。动力学与 Q 资格暂不开放：下一步还需构造并验证动作—响应对齐，Q 另需可靠奖励和终止。原始记录始终保留，不因筛选被删除。

每次尝试使用固定种子的有界 reservoir 抽样，默认示例最多 1000 个同步样本；保留入选数量、候选数与省略数。全快照最多 50000 个样本，逐块处理，像素检查每次只持有当前历史。该数量是软件预算，不是数据足够或模型可驾驶的保证。

快照编码最多 128 MiB，超过预算时在创建输出目录前拒绝。复核报告必须使用新的 `.html` 路径及同名 `.json`，不得覆盖已有文件或写入来源录制目录；例如复核 `dataset.json` 时不要使用 `dataset.html`，因为其 JSON 摘要会与输入重名。

## 覆盖与就绪报告

报告包含每次尝试的连续左右转向、油门/刹车/滑行、速度区间及已核验弯道事件。起步定义为低于 1 km/h 后越过该阈值且伴随油门；松 RT 定义为正纵向输入转为零或负。中断/无效区间切断事件，相邻帧不会各算一次转弯。阈值是版本化的软件统计定义，不能替代赛程能力评估。

补录建议只依据 train/development；evaluation 的行为覆盖不出现在就绪报告中，也不参与补录建议。报告仅展示最终留出组数和样本数；当前数据准备不会使用它调参。`ready_for_software_training` 只说明选中了合格训练和开发样本，`real_candidate_ready` 仍为 false。合成源或未冻结来源标 `diagnostic_only`，不能作为新采集的人工驾驶示范。

## 导出数值历史并训练

`collection-bc-prepare --config configs/collection-bc.example.json --output runs/numeric-candidate` 从固定选择导出数值历史、本车状态、严格因果动作历史及配对参考视图。先替换输入路径与 SHA-256。每帧逐字节复制 RGB 数值数据并按内容去重，不经过 JPEG，也不再次缩放；命名空间保留不同来源的帧身份。沿用 #35 的模型尺寸范围 32–640，每个方向均需满足；超出范围拒绝，而非静默改变已采集分布。独立帧的解码预算共 512 MiB，超限时减小上一步的选择规模。

动作历史取每个偏移截止之前已获得且发生在当前核验区间内的实际输入；没有合格输入时保留缺失 mask。监督标签仍为决策之后的下一轮询，与 actor 的历史分离。本车世界坐标、来源身份、尝试 ID、标签时刻与筛选原因不进入 actor。

默认参考缺失，两种视图的参考 mask 均为空。可选 `reference` 为 `{"route_file":"../runs/independent-route/route.json","independence_evidence":["说明为何独立于当前训练/留出尝试"]}`；先绑定资产哈希，再加载复制后的路线包。禁止与当前采集 session 同源；其他独立性依赖给出的证据，不能仅凭哈希不同就认定独立。参考只生成局部航点，其与无参考视图保持同组，不能把未核实路线当作合法进度依据。

输出 `dataset.json` 只含 train/development，`evaluation.json` 单独保留最终留出。准备报告不显示最终行为或误差。训练配置按 `configs/temporal-bc.example.json` 的字段填写，`dataset` 指向新 `dataset.json`，SHA-256 取准备报告，`time_mode` 使用 `actual`；然后运行 `temporal-train --config <配置> --output <新模型目录>` 和 `temporal-replay --model <模型目录> --dataset <数值数据集> --report <新回放.html>`。模型读取内存中的 RGB 数值，PNG 只用于离线报告预览。参考缺失不声称已经验证有参考驾驶；合成来源始终带有 `diagnostic_only`。

后续工作：独立最终留出评估入口、兼容冻结基线比较，以及真实共享 GPU/CPU 的资源调度验证。当前已经能在后台合成采集继续封存时，用固定快照完成 CPU 训练与加载回放；该原型不代表 4K 游戏负载通过。实际 4K 条件与新人工多次驾驶仍按 #34/#37 验收。
