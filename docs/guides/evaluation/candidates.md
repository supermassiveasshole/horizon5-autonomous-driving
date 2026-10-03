# 候选比较、归档与版本保存

冻结评估之后，先[比较独立证据](#candidate-selection)，再[保留完整 learner](#candidate-archive)并[登记开发版本](#candidate-store)。比较、归档和版本引用的变化均不自动激活真实驾驶控制器。

<a id="candidate-selection"></a>

## 候选比较与晋升边界（T13 / #14）

`candidate-compare` 从两份冻结开发批次及其完整录制清单重新审核，先检查有效性，再比较可靠性与用时。它是版本筛选的软件基础；当前只生成**局部记录的比较建议**，不更新实际默认驾驶版本。已有评估接口仍不能证明自主驾驶、完整独立性和实机收益，因此不能把建议当作晋升。

```powershell
uv run --locked fh5 candidate-compare --config configs/my-comparison.json --output runs/comparison-001 --registry runs/evidence-usage.sqlite
```

配置需先准备，路径相对配置文件解析；登记库可省略，省略时独立性仍未知。公共接口为 `run_experiment(CandidateCompare(config_file, output_dir, registry_file))`。

```json
{
  "version": 1,
  "incumbent": {
    "batch": "../runs/incumbent-batch",
    "batch_sha256": "<batch.json 的 SHA-256>",
    "ledger": "../runs/incumbent-ledger.json",
    "ledger_sha256": "<完整 ledger.json 的 SHA-256>"
  },
  "candidate": {
    "batch": "../runs/candidate-batch",
    "batch_sha256": "<batch.json 的 SHA-256>",
    "ledger": "../runs/candidate-ledger.json",
    "ledger_sha256": "<完整 ledger.json 的 SHA-256>"
  }
}
```

### 冻结规则与证据

沿用[冻结评估](evaluation.md)在录制前绑定的条件、全部运行边界和比较门槛。两批须具有相同车辆/调校/辅助/相机/导航、任务及其依赖文件、像素/时间/动作契约、各参考条件计划次数和 `criteria`。模型可不同；不得在看到结果后为候选降低门槛。`purpose=final` 批次拒绝进入这个开发选模入口，保留独立最终验收用途。

程序重新运行 `EvaluationReview`，不读外部汇总表作为判断依据。核对实际消耗的批次/清单字节及输入依赖，保留原审核结果、全部尝试、未开始槽位和隔离原因。相同非空原始包流不能分别充当两个模型的表现；已登记的跨用途复用也阻止比较。登记库错误不放行；没有已知重叠不等于证明独立。

旧登记库无法还原过去槽位历史时，`legacy_slot_history_unavailable` 阻止局部推荐，避免用成功记录覆盖过去失败。尝试可以同时具有驾驶失败和待核验事项；`pending_checks` 单独保留这些事项，失败分类不能掩盖疑似违规或核验缺口。缺少原始包文件摘要的槽位保留为接口异常，仍输出完整比较报告。

比较顺序固定为：

1. 任一批次存在确认违规、待核验、接口异常、未完成计划、未知尝试数或已知证据复用时，保留既有版本，不产生激进候选建议。驾驶失败仍计入分母。
2. 每批总有效次数须达到既有 `min_valid_attempts`，每个计划参考条件至少有一个正时长的有效完成。这里不将原有总次数门槛偷偷改为每组门槛。
3. 有参考/无参考逐组比较。可靠性使用有效完成数除以全部尝试数，下降不能超过冻结的 `reliability_tolerance`；中位用时不能有任何一组退步，且至少一组的正改善达到 `min_time_improvement_fraction`，才输出 `prefer_candidate_locally`。
4. 可靠性退步超过容差、但合法最快用时改善达到门槛的组，单独列入 `aggressive_by_reference`。这只描述该输入条件，不把不同参考条件混成全局胜利，也不自动激活候选。

容差恰好相等时允许；完赛次数用有理数运算，时间门槛按报告十进制数值比较，避免浮点舍入改变门槛。门槛只是预定的经验规则，不是统计置信保证。可靠性改善但用时变慢时记录取舍，不创造事后加权总分。接触及已核验执行指标保留在两份审核结果中；当前规则未额外引入未经冻结的平顺性阈值。

### 输出与未完成能力

输出新目录包含原配置、两份完整重审结果、`selection.json` 和 `report.html`。`selected_model_sha256` 是局部比较建议的模型身份；`promotion_allowed=false`、`default_changed=false` 始终明确。`promotion_blockers` 解释闭环证据、独立性和诊断来源的缺口。退出码 0 表示比较完成，不能解释为驾驶成功或已经晋升；输入错误退出 2。

程序不修改、删除或续训任何模型，也不清理落选候选。使用[完整候选归档与恢复](#candidate-archive)保留 SAC 续训包；冻结评估中的推理副本不是包含优化器、经验和祖先档案的备份。输出保留证据关联和摘要，依赖原始批次、录制及审核依据，不是所有数据的自包含备份。

需要持久保存比较结果时，继续下文的归档与版本库步骤；比较建议本身不激活驾驶版本。

<a id="candidate-archive"></a>

## 完整候选归档与恢复（T13 / #14）

`candidate-archive` 保留完整 SAC 续训包，`candidate-restore` 将它恢复到一个新目录。原策略清单及权重逐字节保持，因此同一候选仍能关联此前冻结评估；归档本身不创建新的学习阶段、不激活默认驾驶版本，也不授予驾驶资格。

比较后的选择历史与回退由下文版本库维护；实际默认驾驶版本的自动激活和独立实机验收仍未完成，#14 保持开放。

### 使用方式

```powershell
uv run --locked fh5 candidate-archive --checkpoint runs/candidate --checkpoint-sha256 <policy.json摘要> --output runs/archive-001 --reason "保留落选候选继续探索"
uv run --locked fh5 candidate-restore --archive runs/archive-001 --archive-sha256 <上一命令输出的archive_sha256> --output runs/restored-001 --reason "从保留候选继续学习"
uv run --locked fh5 sac-resume --checkpoint runs/restored-001 --output runs/continued-001 --steps 100
```

尖括号是待填写的摘要，不是可直接运行的参数。公共请求为 `CandidateArchive`、`CandidateRestore`，均通过 `run_experiment` 调用。归档返回 JSON，包含档案与原模型摘要、学习状态、累计更新数、保留原因；CLI 不打印完整文件表，文件表保存在 `archive.json` 中。

支持已封存的 SAC 版本 2、3、4，包括示范混合、临时模仿阶段及已退出状态。旧版本 1 和冻结评估中的推理副本缺少完整恢复依赖，不能冒充续训档案。此入口当前沿用已有 CPU/合成经验恢复契约，不代表实机学习已验收。

### 保留范围与核验

- 保存模型、编码器、目标网络、优化器、温度、RNG、累计阶段及教师；不重新初始化 learner。
- 保存续训需要的数值图像、完整转移清单、混合叶来源、历史清单/报告及模仿阶段审核证据。不依赖原训练目录。
- 祖先历史支持旧内嵌数组和新摘要关联节点。归档沿引用逐个保留节点，模型清单保持原字节；恢复后可继续扩展历史，原目录无需存在。存储规划也核验节点摘要并计入依赖。
- 只复制依赖图引用的资产；目录中的其他笔记、临时文件或凭据不随目录递归复制。原始实机录像、原始遥测及外部评估批次应另外保留，此档案不是全部实验素材备份。
- SAC 的可选逐次更新 JSONL 日志不属于恢复依赖，不随档案复制；训练摘要保留日志创建时的状态与内容摘要。需要逐步诊断时另行保留，详见[训练诊断](../sac/training-diagnostics.md)。
- 在私有目录重建候选，核验历史与经验，使用续训共用的初始化入口恢复网络、优化器、温度和 RNG，并执行数值观测推理。恢复后的完整学习状态摘要必须等于原摘要，之后才复制到目标。核验不更新参数、不生成后继训练阶段，也不额外占用续训历史容量。

源摘要、依赖哈希或训练状态有问题时拒绝；已有目标目录不能覆盖，输出不能放在源目录里。归档后可直接从 `archive-001/checkpoint` 继续训练，但通常先恢复到新工作目录更清晰。后续续训依然必须输出到新目录。

### 发布、故障与容量

模型清单在其依赖文件之后写入，归档的 `archive.json` 和恢复的 `restored-from.json` 最后写入。只有操作成功返回且相应清单存在，才能把结果当作完成的档案或恢复；写盘中断留下的目录不能重用或当作成功。原来源不会被删除或改写。

档案不再设置固定文件大小、总大小或依赖条数拒绝阈值。依赖逐件流式复制到私有目录，校验复制字节后恢复 learner；发布到目标时再次验证摘要，并在最后写模型清单前复查目标依赖。核验仍需要临时副本及实际磁盘空间；真实写入失败或依赖不完整时拒绝发布，保留原来源，不自动裁剪历史或丢弃经验。

档案文件清单仍整体载入/生成；其容量治理须按[资源约束](../../design/resource-policy.md)继续推进，不能视作整仓增长结构已清理。版本库的当前读取与容量行为见下文。[审计](../../validation/resource-limit-audit.md)记录剩余范围。

测试包含真实 CPU 更新前后的状态等价、超过旧 256 MiB 的有效报告归档/恢复/续训、原目录移走、不同训练阶段、文件损坏、身份不匹配、目标不可覆盖及写盘故障。软件通过不表示候选驾驶更快或更可靠，实际默认版本仍需要对应独立驾驶证据。

<a id="candidate-store"></a>

## 持久候选版本（T13 / #14）

版本库把默认版本与继续训练的探索版本分开保存，并追加选择和回退历史。每个角色指向[完整 SAC 归档](#candidate-archive)，包括网络、优化器、温度、经验、随机状态和祖先记录。回退默认版本不删除落选候选的训练进度。

版本库支持 `synthetic_development_only` 和 `native_development_only`，分别核验合成执行和原生接口执行的冻结评估、独立起跑、完整尝试与已登记原始数据用途。选择只更新保存的开发版本引用，`default_changed=false`、`real_driving_validated=false` 始终保留；不激活控制器。引用变化分别记录为 `synthetic_default_changed` 或 `native_default_changed`，不代表已证明赛车更强。

### 输入和行为

公共入口是 `run_experiment(CandidateRecord(...))`、`CandidateHistory(...)` 和 `CandidateRollback(...)`。首次记录要求目标目录不存在；后续记录和回退必须提供读取到的当前 `expected_revision`，并保持同一冻结协议。新比较中的 incumbent 必须是版本库当前默认模型。

```powershell
uv run --locked fh5 candidate-record --config configs/my-candidate-record.json --store runs/versions --registry runs/evidence-usage.sqlite
uv run --locked fh5 candidate-history --store runs/versions
uv run --locked fh5 candidate-history --store runs/versions --after-sequence 0 --limit 20
uv run --locked fh5 candidate-history --store runs/versions --limit 0
uv run --locked fh5 candidate-rollback --store runs/versions --expected-revision <当前摘要> --target-revision <历史摘要> --reason "回退说明" --registry runs/evidence-usage.sqlite
```

后续 `candidate-record` 同样传入 `--expected-revision`。命令输出 JSON；退出 0 表示事务或查询完成，不能解释为实机驾驶成功。输入或状态不匹配时退出 2。

记录配置版本 1 选择合成开发范围；版本 2 选择原生开发范围，其他字段相同。路径相对配置文件解析：

```json
{
  "version": 1,
  "comparison": {
    "file": "comparison.json",
    "sha256": "<完整 CandidateCompare 配置的 SHA-256>"
  },
  "checkpoints": {
    "incumbent": "../runs/reliable-checkpoint",
    "candidate": "../runs/exploring-checkpoint"
  }
}
```

`comparison.json` 采用[候选比较](#candidate-selection)的原始批次/清单绑定，不接受外部表格中的 passed 标志。完整源模型身份须与冻结评估相符；评估前后的模型不可以混用。探索版本保存此次候选，即使它未达选择门槛。

原生范围要求版本 3 冻结评估中的 CPU 确定性 SAC 执行逐项通过数值重放，并有完整 UDP 起跑依据、独立有效性审核和相互分离的训练/评估原件。完整候选须来自 `native` 或 `mixed` 经验，父 BC 来自非诊断的 `continuous_numeric_collection`，其已确认输入条件与冻结批次一致。`mixed` 保持混合来源，不改标为纯原生。现任默认也须满足资格，不能以首条登记为由豁免。

一个版本库只能使用同一种范围；不能把已有合成库切换成原生库，反向也不允许。原生软件测试使用真实 CPU 模型和模拟外部设备，只能验证上述流程；实际 FH5 独立驾驶和性能收益仍待验收。

合法且证据完整、最快用时改善但可靠性下降的候选另存为 `aggressive_by_reference`，区分参考输入条件；当前默认保持原样。违规、待核验或缺少执行证据不能新增激进纪录。后续普通候选和默认回退仍保留既有激进角色，其原始选择事件提供评估依据。

### 历史与回退

`state.sqlite` 中只有事务提交后的事件属于有效历史。事件绑定父版本摘要、协议、模型归档和选择理由；源证据审核、归档写入完成后才提交。过期请求或中途写盘失败不能覆盖已提交版本。未提交的工作目录可能留下，既不计入历史，也不自动清理。

历史查询核验事件摘要链及保存的请求/比较报告摘要；查询不重新加载全部模型或宣称源证据仍可用于驾驶资格。SQLite 游标逐条读取，附件流式计算摘要，不设置事件条数、事件大小或数据库大小的人为拒绝阈值；写入按事务追加，真实磁盘/SQLite 接口错误保留已提交状态。完整模型资产及每次操作的临时副本仍需要实际磁盘空间，长期去重与回收另行处理。

`CandidateHistory(store, after_sequence=N, limit=M)` / CLI 同名选项可按需获取历史页。序号从 1 起，返回 N 之后至多 M 条；示例的 20 只控制该次展示数量，不限制版本库继续增长。结果包含 `history_count`、`next_sequence` 和 `history_complete`；下一页使用返回的 `next_sequence`。不传 `limit` 保持完整历史查询行为，会显式物化全部返回条目；长历史应指定页面大小。`limit=0` 仅返回当前角色及统计，训练循环使用此模式，回退仅保留当前事件与目标事件。每页仍逐条验证完整摘要链和附件，页外损坏不会被跳过；累计历史越长，完整校验的 I/O 时间仍会增长。连续分页时应核对顶层 `revision`，若已变化则重新查询，避免把不同版本库快照的页面混用。

核验在私有磁盘快照上进行：通过 SQLite 在线备份按页复制，关闭源连接后再读事件、核验附件，避免慢速诊断长期持有源库读锁而阻止候选提交。一页是 SQLite 的复制粒度，不是数据库容量门槛。此过程需要临时磁盘空间；失败保持原库，不能把失败快照当作成功历史。备份接口报告 `SQLITE_BUSY` / `SQLITE_LOCKED` 时返回明确可重试错误，释放临时快照，待写入方释放锁后可重新查询；不让接口内部无限重试。持续并发写入仍可能让成功的复制步骤重做，尚不保证固定完成时限。[Python backup 接口](https://docs.python.org/3.12/library/sqlite3.html#sqlite3.Connection.backup) · [SQLite 备份锁与一致性](https://www.sqlite.org/backup.html#file_and_database_connection_locking)

回退指定历史事件的默认版本，并重新读取其冻结比较、原始评估证据和完整模型归档；资料缺失或改变时拒绝回退。回退自身追加一个事件，保留当前探索版本与所有中间失败记录。

模型归档可以独立续训，评估证据仍依赖原始批次、录制、审核依据及用途登记库。版本库不是所有实验数据的自包含备份。真实驾驶资格、自动激活及长期磁盘回收仍另行验收；原生短段调度接入见[连续学习循环](../sac/learning-loop.md)，其软件实现不构成真实闭环验收。
