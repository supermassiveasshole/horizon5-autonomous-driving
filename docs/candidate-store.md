# 持久候选版本（T13 / #14）

版本库把默认版本与继续训练的探索版本分开保存，并追加选择和回退历史。每个角色指向[完整 SAC 归档](candidate-archive.md)，包括网络、优化器、温度、经验、随机状态和祖先记录。回退默认版本不删除落选候选的训练进度。

版本库支持 `synthetic_development_only` 和 `native_development_only`，分别核验合成执行和原生接口执行的冻结评估、独立起跑、完整尝试与已登记原始数据用途。选择只更新保存的开发版本引用，`default_changed=false`、`real_driving_validated=false` 始终保留；不激活控制器。引用变化分别记录为 `synthetic_default_changed` 或 `native_default_changed`，不代表已证明赛车更强。

## 输入和行为

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

`comparison.json` 采用[候选比较](candidate-selection.md)的原始批次/清单绑定，不接受外部表格中的 passed 标志。完整源模型身份须与冻结评估相符；评估前后的模型不可以混用。探索版本保存此次候选，即使它未达选择门槛。

原生范围要求版本 3 冻结评估中的 CPU 确定性 SAC 执行逐项通过数值重放，并有完整 UDP 起跑依据、独立有效性审核和相互分离的训练/评估原件。完整候选须来自 `native` 或 `mixed` 经验，父 BC 来自非诊断的 `continuous_numeric_collection`，其已确认输入条件与冻结批次一致。`mixed` 保持混合来源，不改标为纯原生。现任默认也须满足资格，不能以首条登记为由豁免。

一个版本库只能使用同一种范围；不能把已有合成库切换成原生库，反向也不允许。原生软件测试使用真实 CPU 模型和模拟外部设备，只能验证上述流程；实际 FH5 独立驾驶和性能收益仍待验收。

合法且证据完整、最快用时改善但可靠性下降的候选另存为 `aggressive_by_reference`，区分参考输入条件；当前默认保持原样。违规、待核验或缺少执行证据不能新增激进纪录。后续普通候选和默认回退仍保留既有激进角色，其原始选择事件提供评估依据。

## 历史与回退

`state.sqlite` 中只有事务提交后的事件属于有效历史。事件绑定父版本摘要、协议、模型归档和选择理由；源证据审核、归档写入完成后才提交。过期请求或中途写盘失败不能覆盖已提交版本。未提交的工作目录可能留下，既不计入历史，也不自动清理。

历史查询核验事件摘要链及保存的请求/比较报告摘要；查询不重新加载全部模型或宣称源证据仍可用于驾驶资格。SQLite 游标逐条读取，附件流式计算摘要，不设置事件条数、事件大小或数据库大小的人为拒绝阈值；写入按事务追加，真实磁盘/SQLite 接口错误保留已提交状态。完整模型资产及每次操作的临时副本仍需要实际磁盘空间，长期去重与回收另行处理。

`CandidateHistory(store, after_sequence=N, limit=M)` / CLI 同名选项可按需获取历史页。序号从 1 起，返回 N 之后至多 M 条；示例的 20 只控制该次展示数量，不限制版本库继续增长。结果包含 `history_count`、`next_sequence` 和 `history_complete`；下一页使用返回的 `next_sequence`。不传 `limit` 保持完整历史查询行为，会显式物化全部返回条目；长历史应指定页面大小。`limit=0` 仅返回当前角色及统计，训练循环使用此模式，回退仅保留当前事件与目标事件。每页仍逐条验证完整摘要链和附件，页外损坏不会被跳过；累计历史越长，完整校验的 I/O 时间仍会增长。连续分页时应核对顶层 `revision`，若已变化则重新查询，避免把不同版本库快照的页面混用。

核验在私有磁盘快照上进行：通过 SQLite 在线备份按页复制，关闭源连接后再读事件、核验附件，避免慢速诊断长期持有源库读锁而阻止候选提交。一页是 SQLite 的复制粒度，不是数据库容量门槛。此过程需要临时磁盘空间；失败保持原库，不能把失败快照当作成功历史。备份接口报告 `SQLITE_BUSY` / `SQLITE_LOCKED` 时返回明确可重试错误，释放临时快照，待写入方释放锁后可重新查询；不让接口内部无限重试。持续并发写入仍可能让成功的复制步骤重做，尚不保证固定完成时限。[Python backup 接口](https://docs.python.org/3.12/library/sqlite3.html#sqlite3.Connection.backup) · [SQLite 备份锁与一致性](https://www.sqlite.org/backup.html#file_and_database_connection_locking)

回退指定历史事件的默认版本，并重新读取其冻结比较、原始评估证据和完整模型归档；资料缺失或改变时拒绝回退。回退自身追加一个事件，保留当前探索版本与所有中间失败记录。

模型归档可以独立续训，评估证据仍依赖原始批次、录制、审核依据及用途登记库。版本库不是所有实验数据的自包含备份。真实驾驶资格、自动激活及长期磁盘回收仍另行验收；#15 连续调度的原生接入尚未因此完成。
