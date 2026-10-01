# 持久候选版本（T13 / #14）

版本库把默认版本与继续训练的探索版本分开保存，并追加选择和回退历史。每个角色指向[完整 SAC 归档](candidate-archive.md)，包括网络、优化器、温度、经验、随机状态和祖先记录。回退默认版本不删除落选候选的训练进度。

当前资格范围为 `synthetic_development_only`：实际运行 CPU 策略与可响应动作的合成环境，重新核验冻结评估的执行记录、独立起跑、完整尝试与已登记原始数据用途。没有原生 FH5 激活路径，`default_changed=false`、`real_driving_validated=false` 始终保留。合成默认版本变化单独标为 `synthetic_default_changed`；不能据此宣称赛车更强。

## 输入和行为

公共入口是 `run_experiment(CandidateRecord(...))`、`CandidateHistory(...)` 和 `CandidateRollback(...)`。首次记录要求目标目录不存在；后续记录和回退必须提供读取到的当前 `expected_revision`，并保持同一冻结协议。新比较中的 incumbent 必须是版本库当前默认模型。

```powershell
uv run --locked fh5 candidate-record --config configs/my-candidate-record.json --store runs/versions --registry runs/evidence-usage.sqlite
uv run --locked fh5 candidate-history --store runs/versions
uv run --locked fh5 candidate-rollback --store runs/versions --expected-revision <当前摘要> --target-revision <历史摘要> --reason "回退说明" --registry runs/evidence-usage.sqlite
```

后续 `candidate-record` 同样传入 `--expected-revision`。命令输出 JSON；退出 0 表示事务或查询完成，不能解释为实机驾驶成功。输入或状态不匹配时退出 2。

记录配置为版本 1，路径相对配置文件解析：

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

合法且证据完整、最快用时改善但可靠性下降的候选另存为 `aggressive_by_reference`，区分参考输入条件；当前默认保持原样。违规、待核验或缺少执行证据不能新增激进纪录。后续普通候选和默认回退仍保留既有激进角色，其原始选择事件提供评估依据。

## 历史与回退

`state.sqlite` 中只有事务提交后的事件属于有效历史。事件绑定父版本摘要、协议、模型归档和选择理由；源证据审核、归档写入完成后才提交。过期请求或中途写盘失败不能覆盖已提交版本。未提交的工作目录可能留下，既不计入历史，也不自动清理。

历史查询核验事件摘要链及保存的请求/比较报告摘要；查询不重新加载全部模型或宣称源证据仍可用于驾驶资格。历史最多 1000 个事件、每事件 4 MiB，数据库限 32 MiB。完整模型资产另受归档容量约束，且每次操作需要额外归档空间；长期去重与回收由资源管理切片处理。

回退指定历史事件的默认版本，并重新读取其冻结比较、原始评估证据和完整模型归档；资料缺失或改变时拒绝回退。回退自身追加一个事件，保留当前探索版本与所有中间失败记录。

模型归档可以独立续训，评估证据仍依赖原始批次、录制、审核依据及用途登记库。版本库不是所有实验数据的自包含备份。未知转换谱系、实际驾驶资格、原生 FH5 晋升及长期磁盘回收仍另行验收。
