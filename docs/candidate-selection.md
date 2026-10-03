# 候选比较与晋升边界（T13 / #14）

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

## 冻结规则与证据

沿用[冻结评估](evaluation.md)在录制前绑定的条件、全部运行边界和比较门槛。两批须具有相同车辆/调校/辅助/相机/导航、任务及其依赖文件、像素/时间/动作契约、各参考条件计划次数和 `criteria`。模型可不同；不得在看到结果后为候选降低门槛。`purpose=final` 批次拒绝进入这个开发选模入口，保留独立最终验收用途。

程序重新运行 `EvaluationReview`，不读外部汇总表作为判断依据。核对实际消耗的批次/清单字节及输入依赖，保留原审核结果、全部尝试、未开始槽位和隔离原因。相同非空原始包流不能分别充当两个模型的表现；已登记的跨用途复用也阻止比较。登记库错误不放行；没有已知重叠不等于证明独立。

旧登记库无法还原过去槽位历史时，`legacy_slot_history_unavailable` 阻止局部推荐，避免用成功记录覆盖过去失败。尝试可以同时具有驾驶失败和待核验事项；`pending_checks` 单独保留这些事项，失败分类不能掩盖疑似违规或核验缺口。缺少原始包文件摘要的槽位保留为接口异常，仍输出完整比较报告。

比较顺序固定为：

1. 任一批次存在确认违规、待核验、接口异常、未完成计划、未知尝试数或已知证据复用时，保留既有版本，不产生激进候选建议。驾驶失败仍计入分母。
2. 每批总有效次数须达到既有 `min_valid_attempts`，每个计划参考条件至少有一个正时长的有效完成。这里不将原有总次数门槛偷偷改为每组门槛。
3. 有参考/无参考逐组比较。可靠性使用有效完成数除以全部尝试数，下降不能超过冻结的 `reliability_tolerance`；中位用时不能有任何一组退步，且至少一组的正改善达到 `min_time_improvement_fraction`，才输出 `prefer_candidate_locally`。
4. 可靠性退步超过容差、但合法最快用时改善达到门槛的组，单独列入 `aggressive_by_reference`。这只描述该输入条件，不把不同参考条件混成全局胜利，也不自动激活候选。

容差恰好相等时允许；完赛次数用有理数运算，时间门槛按报告十进制数值比较，避免浮点舍入改变门槛。门槛只是预定的经验规则，不是统计置信保证。可靠性改善但用时变慢时记录取舍，不创造事后加权总分。接触及已核验执行指标保留在两份审核结果中；当前规则未额外引入未经冻结的平顺性阈值。

## 输出与未完成能力

输出新目录包含原配置、两份完整重审结果、`selection.json` 和 `report.html`。`selected_model_sha256` 是局部比较建议的模型身份；`promotion_allowed=false`、`default_changed=false` 始终明确。`promotion_blockers` 解释闭环证据、独立性和诊断来源的缺口。退出码 0 表示比较完成，不能解释为驾驶成功或已经晋升；输入错误退出 2。

程序不修改、删除或续训任何模型，也不清理落选候选。使用[完整候选归档与恢复](candidate-archive.md)保留 SAC 续训包；冻结评估中的推理副本不是包含优化器、经验和祖先档案的备份。输出保留证据关联和摘要，依赖原始批次、录制及审核依据，不是所有数据的自包含备份。

[持久候选版本](candidate-store.md)将合成或原生接口的策略执行、起跑与原始证据重审接入各自独立的开发版本库，分别保存默认、探索与激进角色，并支持追加历史及回退。它保留本比较器的实际晋升限制，不将诊断记录变成 FH5 驾驶资格。#14 仍未完成实际默认版本的自动激活和真实候选的独立驾驶验证。
