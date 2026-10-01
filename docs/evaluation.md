# 冻结评估与全部尝试统计（T09 / #10）

当前提供局部任务的离线评估协议、批次回放和重复执行入口。版本 1 冻结数值 Δt BC，版本 2 显式冻结 SAC，版本 3 为原生 BC 执行额外绑定数值采集条件和推理设备；均固定权重、视觉编码器、预处理和历史契约，复用独立的[尝试有效性判定](attempts.md)保留全部结果。`evaluation-prepare` / `evaluation-review` 不采集、不操作游戏、不训练策略。

## 准备批次

```powershell
uv run --locked fh5 evaluation-prepare --config configs/evaluation.example.json --output runs/evaluation-001
```

先安装 `learning` 可选依赖。替换示例中的模型目录、局部任务文件及 SHA-256；路径相对配置文件。示例门槛只是候选值，不是已验收标准。固定配置包括车辆/调校/辅助快照、相机与导航、独立任务依据、像素/时间契约、决策频率/期限/动作有效期、各次计划的参考模式，以及最少有效次数和比较容差。`exploration`、`rewind` 必须为 false；配置表示评估禁用倒带，不声称已经修改游戏开关。

版本 1 模型必须是受支持的数值 Δt BC，并实际训练过计划中的每种参考条件。运行配置必须满足模型像素/历史尺寸和任务车辆条件；这不验证实际驾驶输入。输出复制 `model.json`、`actor.pt`、局部任务及路线/走廊/检查点和依据文件，并由 `batch.json` 绑定哈希。不会读取开发目录里的后续改动。输出使用新目录；未写出 `batch.json` 的中途失败目录不能用作已冻结批次。

必须在计划运行前准备；现有记录事后重算只算诊断。早于冻结时刻的录制、时间无效/无时区或先后顺序不明的录制保留原局部判定，但不能贡献本批有效成绩。`protocol_order` 只核对会话与批次声明的 UTC 时间，不证明实际执行了冻结策略；时间错误须回到原始证据核实，不能改写旧时间使其合格。`purpose: development | final` 保存预定用途；最终批次预登记与已知数据复用检查见下文，仅填 `final` 不构成独立最终验收。

## 回放清单

录制结束后创建 UTF-8 JSON 清单。`batch_sha256` 是上述 `batch.json` 的 SHA-256；`files` 必须列出每份源录制顶层全部 `.json` / `.jsonl` 文件及哈希，包括存在的控制、策略或赛事日志，不能只挑成功片段。`evidence` 使用 `attempt-review` 的独立审核格式，缺少时填 null。路径相对清单文件。

```json
{
  "version": 1,
  "batch_sha256": "<冻结 batch.json 的 SHA-256>",
  "entries": [
    {
      "slot_id": "no-reference-1",
      "recording": "../recording-001",
      "files": {
        "session.json": "<SHA-256>",
        "packets.jsonl": "<SHA-256>",
        "report.json": "<如存在则列出 SHA-256>"
      },
      "evidence": {"file": "review/evidence.json", "sha256": "<SHA-256>"}
    }
  ]
}
```

```powershell
uv run --locked fh5 evaluation-review --batch runs/evaluation-001 --ledger runs/evaluation-ledger.json --output runs/evaluation-review-001
```

输出 `batch-report.json`、静态 `report.html` 及各录制的独立有效性回放。清单一个槽位对应一份完整录制；确认重开拆出的额外尝试全部保留。重复槽位、复制的非空包流或冻结文件改变会拒绝整个清单；空包流配合不同会话仍保留为各自的接口异常。此去重不证明数据从未用于训练，也不识别任意重编码后的复本。

源录制和审核依据在回放前后核验。损坏或无法读取的录制保留一个 `interface_error` 占位，说明实际尝试数未知；其他录制继续处理。退出码 0 表示清单已解析，2 表示输入失败、未解析录制或提供的执行证据未通过核验，并不表示驾驶成功或失败。保留原录制和证据目录，批次报告不是完整原始数据备份。

## 绑定数值执行证据

清单每个条目可以增加 `execution`，不提供时保持旧的局部有效性回放行为：

```json
"execution": {
  "directory": "../realtime-run-001",
  "manifest_sha256": "<realtime-manifest.json 的 SHA-256>"
}
```

当前接受实时 v2 的合成、只读影子或版本 3 批次绑定的原生 BC 记录，使用冻结模型独立重载数值输入和预测；兼容标准数值 Δt actor 及匹配像素契约的适配器。原生记录须与冻结数值条件和实际 worker 推理设备一致。回放不会重新发送动作，也不把影子输出升级为实际驾驶证据。

核验配置、模型、图像和时序存档、日志完整性，以及整个原始遥测包流与该录制完全一致。逐决策核对本车状态、遥测年龄、安全状态和游戏时钟；实际参考掩码必须符合计划模式。合成及原生动作历史按冻结偏移和实时 v2 的 200 ms 查找容差，从成功发送日志重建；影子记录保持未知动作历史，不把未发往游戏的建议当成先前动作。成功命令须与重放预测、幅度包络、期限相符，worker 执行、结果返回和发送必须顺序因果，最后必须有解除输入记录。此机制检查保存证据的一致性，不证明传感器、标注或声明来源本身真实。

`executions` 每个录制仅一条，避免重开拆出多个尝试后重复累计运行时长。状态为 `missing`、`source_unreadable`、`quarantined` 或 `bound_diagnostic`。提供的证据不匹配时，原局部结论保留，局部成功转为本批待核验；驾驶失败与已确认违规不会被覆盖。存档缺口和核验错误均列出，隔离记录仍在尝试分母内，但不进入已核验执行指标。

通过核验的诊断执行报告如下：

- 有效决策率为接受的决策数除以运行窗口秒数；批次按总次数／总时长计算，不平均各轮频率。最长跳过时长由失败决策起点到下一接受决策或运行结束计算。
- 图像年龄来自源帧与决策时间；视觉比例为“具有完整且已记录 actor 图像输入的决策／全部调度决策”。跳过期间画面是否可用可能未知，此比例不是 DXGI 丢帧率。
- 控制权按成功发送返回之间的时长统计，首次发送前计为 `unobserved`，解除输入和租期到期单列。hold 使用发送返回时刻作为代理，不能解释为游戏实际收到动作的精确时间。
- 转向总变化量和变化方向反转次数只统计相邻策略命令，中间有监督器动作则断开；不据此直接判定“无效振荡”，正常转弯也会改变方向。
- `execution_metrics_by_observed_reference` 按已记录 actor 的实际参考模式归组；原 `by_reference` 仍按冻结计划统计尝试，不能混用。各次保留合成／影子来源，两者均禁止晋升。

图像与动作的独立重放诊断保存在 `execution-NNNN.html/json`；缺文件等情况可能只有批次中的错误项。`recorded_gaps` 是绑定哈希后的原记录声明，成功核验另由 `status` 表示，不能只看“丢弃为零”就判完整。

## 如何解读

- 分别列出有效完成、驾驶失败、违规、待核验及接口异常；缺证据仍保留，未开始的计划另列，不伪造成驾驶尝试。
- `valid_fraction_all_attempts` 以所有已解析尝试及异常占位为分母；存在 `unresolved_recordings` 时无法声称掌握完整真实尝试数。另列排除待核验/接口异常后的 `valid_fraction_classified_driving`，两者不能混用。
- 用时只汇总符合冻结条件的有效连续局部片段，列出数量、最短、中位和最长；接触计数独立保留。快照不符记为本批 `invalid`；事后绑定或时间不明的局部成功记为本批 `pending_review`，两者均不贡献有效比例/用时，原独立判定另存 `local_outcome` / `local_record_eligible`。已有驾驶失败、违规和接口异常不会被时间缺口抹去。未完赛片段不混入用时分布，恢复后的片段不拼成完整成绩。
- 有参考/无参考尝试分组来自冻结计划；可选执行证据另核对已记录的数值 actor 输入。实际游戏控制与响应、完整视觉来源和自动重开仍有缺口。`valid_complete` 仍是局部有效性结果，不能解释为冻结策略自主成功。
- 所有报告均禁止自动晋升；合成数据或诊断模型另有 `diagnostic_only` 标记。冻结阈值由[候选比较入口](candidate-selection.md)用于局部记录筛选，建议与实际默认版本激活分别处理。

## 合成环境中的重复执行

`run_experiment(EvaluationRun(...), evaluation_environment=...)` 将事件准备、冻结数值策略和上述评估串联。入口来自 `fh5.evaluation_run`，参数依次为批次目录、`batch.json` SHA-256、赛事配置、输出目录及每次 `seconds`。环境提供 `event(slot_id)`、`driving(slot_id, ready_state)` 和 `close()`；前两个分别返回现有事件与实时环境接口。它们依次持有输入资源，事件环境关闭后才打开策略环境；合成环境默认运行，原生适配须满足下节额外条件并显式启用 `live`。

运行前复制冻结模型、任务、赛事设置、菜单模板和条件依据，并保存运行时长及文件哈希。第一轮使用开始菜单配方，后续使用重开配方；准备阶段要求两张新鲜匹配画面、原车型/PI、指定起点附近的停车遥测，并且画面和遥测必须晚于最后一次菜单动作。此阶段不冒称已经开始驾驶尝试。策略端从首次新遥测直到首条非零命令，持续核对位置、低起步速度、中立输入和向前推进的游戏时钟；状态变化或交接期限耗尽则拒绝非零命令。

每次启动独立的冻结 actor 和空动作历史；没有 learner、探索或倒带。重开不确认、资源未释放、关键记录不完整或非正常策略停止时，批次终止并列出未开始槽位。已请求启动但打不开驱动也保留为接口尝试。模型或菜单资产被修改则停止，保留原始目录和清单；最终评估无法完成时，运行报告显式保存错误，不伪造通过结果。

外部菜单／驾驶工厂返回后、发送动作前再次核验菜单配置。起点状态传给驾驶工厂的是副本，交接使用原先冻结的事件参数，适配器不能改写校验基准。actor 在读取实际模型清单字节时核对预先绑定的摘要，随后核对加载权重，不能在交接期间替换为另一份合法模型。

输出包含 `run-protocol.json`、`run.json`、冻结副本、各次 `ready/` 和 `execution/`、重建的完整遥测录制、`ledger.json`、`review/` 与入口 `report.html`。当前限制为 1–10 次无参考运行，每次 0.1–600 秒；它检查调用与证据流程，不模拟真实车辆动力学，也不证明自主驾驶。准备阶段和运行上限均有界，但外部 I/O 实现仍须遵守接口的及时返回约定，不能保证强杀进程等情况下的资源释放。

显式自动起跑任务现在可绑定并独立重算准备证据，见下文；旧 `manual_placement` 任务仍保留原语义。参考辅助运行、游戏响应凭据、实机起点交接及完成页重开编排仍需后续实现或验收。用途登记见下文，训练入口的自动登记及完整来源覆盖仍需接入。#4/#9 的实机门槛保留；不阻塞这些独立软件工作。

## 原生 BC 重复评估入口

版本 3 沿用完整评估配置，把 `model` 写为 `{"directory":"…","manifest_sha256":"…","kind":"bc","device":"cuda"}`（也支持 `cpu`），并将模型来源中的 `provenance.input_conditions` 原样放入 `conditions.numeric_input_conditions`。它绑定 4K/其他源尺寸、HUD、相机和数值像素等实际采集条件；不能把旧诊断模型改标签作为资格证明。任务必须为版本 2 的自动起跑任务，计划限 1–10 次无参考运行，每次最多 30 秒。

先用 `evaluation-prepare` 冻结版本 3 批次，再检查：

```powershell
uv run --locked fh5 evaluation-run --batch runs/evaluation-001 --batch-sha256 <batch.json的SHA256> --event-config configs/event.local.json --driving-config configs/numeric-drive.local.json --output runs/evaluation-run-001 --seconds 15
```

默认只验证文件，不打开设备、不创建运行目录。`--driving-config` 使用 [realtime-drive](realtime-decisions.md) 的已绑定配置，必须包含该模型及同一推理设备的原生 DXGI 只读时效证据；模型、运行参数、车辆、路线和采集条件须与批次一致。准备批次和离线检查不需要 CUDA 可用；实际 worker 必须使用冻结设备。首次已在赛事中的停车状态可声明 `--initial-operation restart_ready`，默认 `start_ready` 对应赛前开始菜单。

在实机准备完成且运行已获授权时，追加 `--live` 才执行。菜单阶段直接将 DXGI BGRA 数值像素在内存中缩小为灰度模板输入，使用源帧时间并拒绝重复帧和窗口边界变化，没有 JPEG 热路径。事件环境先释放控制器、采集器和 UDP，再创建数值驾驶适配器；采集线程未退出、关闭出错或设备取得后的清理状态不明，均停止交接并报告释放未确认。沿用 F8、失焦、断流、动作有效期及起点交接检查。每次重新核对原配置，改变条件则停止，不在批次内换模型。

保存全部已启动尝试、未启动槽位、菜单/驾驶输入统计和资源释放结果。版本 3 的候选比较也包含推理设备，CPU 与 CUDA 条件不能混作同一批次对照。退出码 0 仅表示计划执行完且保存证据通过核验，不表示有效完赛；报告仍为诊断结果，不自动晋升候选。`completed_evaluation` 的学习循环恢复目前仍限合成协议，原生批次尚未接入 #15 的恢复路径；原生 SAC 采样/评估另由 #11 推进。

版本 3 不接受旧合成执行入口或合成来源的执行证据，避免把默认 CPU 运行绑定为声明的 CUDA 条件；版本 1/2 仍用于合成验证。本入口当前仅有模拟外部设备的软件证据，详见 [验证记录](validation/t09-native-evaluation.md)。实际 DXGI 菜单匹配、资源切换耗时、游戏响应和真实重复驾驶均未据此验收。

## 自动局部起跑的证据

任务文件可使用 `version: 2`、`control_owner: "policy"`、`start_mode: "automatic_event_ready"`，并增加以下字段；其他路线、车辆和期限字段不变。任务版本与 BC/SAC 批次版本独立。

```json
"automatic_start": {
  "event_file": "event.json",
  "event_sha256": "<赛事配置文件的 SHA-256>",
  "handoff_timeout_s": 5
}
```

交接期限必须大于零、不超过 30 秒，示例 5 秒不是实机验收值。`evaluation-prepare` 校验赛事条件与任务一致，并把菜单模板、条件依据和规范化赛事配置一并冻结至 `start/`。之后修改原配置不能改变此批次。执行入口额外要求传入赛事配置与冻结内容相同。

`EvaluationRun` 为每次准备保存原始遥测、PGM 画面、采集时间、菜单动作发出/返回时间和解除输入记录。输入释放后写入 `ready/start-manifest.json`，绑定批次、槽位、该次完整驾驶包流和数值执行清单；`ledger.json` 的 `preparation` 保存目录与清单摘要。不能用上一轮就绪记录顶替下一轮。

独立 `evaluation-review` 从实际像素重新匹配菜单配方，核对每步操作前两张不同的新鲜画面、最后菜单动作之后的两张驾驶画面与停车遥测，并核对就绪之后的成功解除日志，不只相信汇总旗标。随后检查首个驾驶包至首个成功策略命令之间的全部已收遥测：原车、位置、中立输入、时间顺序、最后遥测年龄及交接期限。画面和准备遥测的年龄上限为 500 ms；这是有界菜单确认的规则，与实时 actor 图像契约分别处理。运行时和独立审核都会检查交接是否过期。

报告 `starts` 和 `verified_starts` 区分 `not_required`、`verified`、`quarantined`。其中 `first_policy_command_ns` 指首条成功非零策略命令；初始中立命令不结束检查窗口，只有中立命令的记录不认证自动起跑。通过仅移除对应首次尝试的 `automatic_start_unverified` 缺口；墙壁、捷径、接管、条件和几何证据仍各自检查。缺失、改写、错槽或超时的准备记录不能得到有效自动起跑，尝试不从分母消失。单独 `attempt-review` 没有完整批次/执行绑定，自动起跑仍待核验。

自动起跑使用 `local-validity-v3`，人工置位继续使用 v2；旧结果不追认。`valid_complete` 和 `record_eligible` 仍只表示局部审核结果，`unattended`、`closed_loop_validated`、自动晋升及全程完赛能力不由此放行。当前运行适配器仅为合成来源，实际验证与局限见[自动起跑记录](validation/t09-automatic-start.md)。

## 最终批次预登记与已知数据复用

同一项目使用一个持久登记库，例如 `runs/evidence-usage.sqlite`。`evaluation-prepare --registry ...` 将批次摘要、冻结模型摘要、用途和登记时间存入库，并把库的身份写入冻结批次；录制应在预登记后开始。复制或移动批次后仍须使用同一个登记库。创建一个新库不能证明旧历史已不存在。

```powershell
uv run --locked fh5 evidence-use --registry runs/evidence-usage.sqlite --role training --recording runs/human-001 --output runs/use-human-001
uv run --locked fh5 evaluation-prepare --config configs/my-evaluation.json --registry runs/evidence-usage.sqlite --output runs/final-001
uv run --locked fh5 evaluation-review --batch runs/final-001 --ledger runs/final-ledger.json --registry runs/evidence-usage.sqlite --output runs/final-review-001
```

示例中的配置和录制需先准备；`--role selection` 表示该录制用于开发／候选选择，`--recording` 可重复，`--model-sha256` 可记录所关联模型清单摘要。登记只记录声明用途和原始文件指纹，不运行训练，也不证明声明完整。后续训练和选择入口须调用这一边界，不能依赖人永久手动补账。

`EvaluationPrepare`、`EvaluationReview`、`EvaluationRun` 均可传 `registry_file`；`RecordUsage` 通过同一个实验入口登记训练／选择用途。审核时自动将开发批次记为 `selection`，最终批次记为 `final`。相同批次的相同用途幂等；后来登记的其他用途会在下次审核时发现。SQLite 事务保留原历史，接口不提供删除或改写旧用途操作。

同一批次中，每个已进入审核的槽位绑定其完整源文件清单摘要；后续审核若删除该槽位或换一份录制，会在生成新报告前拒绝，旧报告和登记继续保留。可以补齐此前尚未开始的槽位，也可以更新同一份原始录制的独立审核证据；不能把一次失败的原始输入换成新成功输入。此约束不要求输入路径相同，移动未改内容的原录制仍可回放。

旧版登记库会在事务中升级至版本 2，保留身份、用途、预登记和已有槽位绑定。若旧库从未保存槽位绑定，全部既有预登记批次及用途中出现的批次均标记 `legacy_slot_history_unavailable`：旧库无法区分尚未开始与已审核但数据为空的尝试，后来补登不能恢复当时缺失的历史。仍可查询旧用途冲突并登记新批次，无需丢弃或重建旧库。

审核将独立性作为单独字段保留，不改写局部驾驶结论或尝试分母：

- `known_overlap`：会话或非空包流的完整 SHA-256 与已登记训练、选模或另一最终批次重叠。改目录名或只改会话注释不能洗掉相同包流的已知用途。
- `no_known_overlap`：批次在同一库预登记、录制时间元数据在登记之后、计划记录齐全且均可辨认，没有发现已登记重叠。它只是必要条件；`independence_proven` 仍为 false，不证明整个训练谱系完整，也不允许晋升。
- `unknown`：未预登记、身份不符、时间不明／未来时间、未开始槽位、空或不可读录制、登记库故障。原录制与尝试统计仍保留。
- `untracked`：未传登记库，兼容旧操作，但不能把 `purpose=final` 标签当作独立性证据。

完整登记快照及其摘要保存为 `usage-snapshot.json`；报告只描述该快照时刻。备份登记库及快照，不能只保存 HTML。该机制不识别任意重新编码、修改时间戳或裁剪后的数据复本，也无法发现从未登记的使用；实际时间顺序目前仍基于录制元数据，不能替代实机执行证据。单次登记上限 1000 份录制，库上限 20000 条指纹用途、20000 个已审核槽位与 10000 个预登记批次。当前登记接口面向含 `session.json` / `packets.jsonl` 的完整遥测录制；持续数值采集和模型数据集的完整血缘仍需适配，不能假定已自动覆盖。

`evidence-use` 有无法辨认的空记录时退出 4；登记库／输入错误退出 2。最终批次审核发现重叠或未知时退出 4，输入／登记库错误退出 2；未传库的旧审核退出语义保持不变。退出 0 也不表示独立性已证实或驾驶成功。

## 冻结 SAC 的版本 2 批次

沿用上述完整配置，将顶层 `version` 设为 `2`，并显式声明模型类型及 `policy.json` 摘要：

```json
"model": {
  "kind": "sac",
  "directory": "runs/sac-candidate",
  "manifest_sha256": "<policy.json 的 SHA-256>"
}
```

模型目录相对配置文件解析。当前要求封存版本 2 的 CPU SAC 候选；批次复制 `policy.json`、`policy.pt`、训练报告及 `bc/` 中的标准化/历史元数据和初始化权重。SAC 编码器和策略实际来自 `policy.pt`，不会退回 BC 驾驶输出。所有运行依赖由批次清单绑定，读取时核对实际字节；该评估副本不包含完整经验和祖先档案，不是续训包。

每种参考条件须存在于初始化的已训练条件中。像素/历史必须一致，动作历史采用 200/100/0 ms；执行器幅度须与候选训练包络完全一致。当前实时配置的油门硬上限仍为 0.25，默认 SAC 的 0.35 包络不能静默裁成 0.25 来比较；需要按可执行契约生成兼容候选，或另行实现并验证扩展执行范围。不能改写旧模型清单冒充兼容。

`EvaluationRun` 复用同一个菜单确认、起点复核、实时线程和全量尝试账本，关闭探索与倒带；每轮重新加载完整冻结 SAC。独立部署的 learner 不参与这个执行入口。该适配器当前只允许合成执行器，不能改标签后接入游戏。来源一律为诊断，任何合成结果均不得自动晋升。

SAC 的动作区间依赖成功命令及其经过时间。首次就绪先发送并记录 `initial_neutral`，成功后下一次决策才计算策略；发送失败不假设中立成功。每个决策单独保存 `command_context`，包含命令编号、整数值、owner 及发送起止时刻；它不混入 BC 的特征字段。实际时间采用决策时刻减上次成功发送返回时刻，仍是主机代理，不声称测得游戏应用时刻。正常推理使用零噪声与冻结参数。

如果推理期间动作租期届满、监督器发出中立命令，旧结果标记 `discard_command_context_changed`，不能沿用旧动作区间发送。后续新决策使用最新成功中立命令；每次重开另建状态和历史。确定性故障回放可指定 `RealtimeReplay(..., require_command_context=True)` 检查这一分支。

数值回放核对存档中的相同上下文和预测；批次审核进一步从成功命令日志独立重建上下文。可重放但没有真实命令来源的上下文仍被隔离，全部尝试保留。用途登记的模型身份绑定 SAC 清单，而非其初始化 BC 清单。

```powershell
uv run --locked fh5 realtime-replay runs/evaluation/attempt-0000/execution --model runs/evaluation/frozen/model --report runs/sac-execution-replayed.html
```

SAC 回放当前使用 CPU 和原数值来源契约，不接受旧来源诊断放行参数。此命令没有环境适配器，不重新发送动作。真实控制与响应、实机自动起跑、参考辅助执行和完整赛程评估仍须后续实现或验收。
