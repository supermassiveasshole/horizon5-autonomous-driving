# 冻结评估与全部尝试统计（T09 / #10）

当前提供局部任务的离线评估协议与批次回放。冻结数值 Δt BC 的权重、视觉编码器、预处理和历史契约，复用独立的[尝试有效性判定](attempts.md)保留全部结果。两个命令均不采集、不操作游戏、不训练策略。

## 准备批次

```powershell
uv run --locked fh5 evaluation-prepare --config configs/evaluation.example.json --output runs/evaluation-001
```

先安装 `learning` 可选依赖。替换示例中的模型目录、局部任务文件及 SHA-256；路径相对配置文件。示例门槛只是候选值，不是已验收标准。固定配置包括车辆/调校/辅助快照、相机与导航、独立任务依据、像素/时间契约、决策频率/期限/动作有效期、各次计划的参考模式，以及最少有效次数和比较容差。`exploration`、`rewind` 必须为 false；配置表示评估禁用倒带，不声称已经修改游戏开关。

模型必须是受支持的数值 Δt BC，并实际训练过计划中的每种参考条件。运行配置必须满足模型像素/历史尺寸和任务车辆条件；这不验证实际驾驶输入。输出复制 `model.json`、`actor.pt`、局部任务及路线/走廊/检查点和依据文件，并由 `batch.json` 绑定哈希。不会读取开发目录里的后续改动。输出使用新目录；未写出 `batch.json` 的中途失败目录不能用作已冻结批次。

必须在计划运行前准备；现有记录事后重算只算诊断。早于冻结时刻的录制、时间无效/无时区或先后顺序不明的录制保留原局部判定，但不能贡献本批有效成绩。`protocol_order` 只核对会话与批次声明的 UTC 时间，不证明实际执行了冻结策略；时间错误须回到原始证据核实，不能改写旧时间使其合格。`purpose: development | final` 保存预定用途，目前还没有最终批次的独立登记与防复用机制，因此仅填 `final` 不构成独立最终验收。

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

源录制和审核依据在回放前后核验。损坏或无法读取的录制保留一个 `interface_error` 占位，说明实际尝试数未知；其他录制继续处理。退出码 0 表示清单已解析，2 表示输入失败或存在未解析录制，并不表示驾驶成功或失败。保留原录制和证据目录，批次报告不是完整原始数据备份。

## 如何解读

- 分别列出有效完成、驾驶失败、违规、待核验及接口异常；缺证据仍保留，未开始的计划另列，不伪造成驾驶尝试。
- `valid_fraction_all_attempts` 以所有已解析尝试及异常占位为分母；存在 `unresolved_recordings` 时无法声称掌握完整真实尝试数。另列排除待核验/接口异常后的 `valid_fraction_classified_driving`，两者不能混用。
- 用时只汇总符合冻结条件的有效连续局部片段，列出数量、最短、中位和最长；接触计数独立保留。快照不符记为本批 `invalid`；事后绑定或时间不明的局部成功记为本批 `pending_review`，两者均不贡献有效比例/用时，原独立判定另存 `local_outcome` / `local_record_eligible`。已有驾驶失败、违规和接口异常不会被时间缺口抹去。未完赛片段不混入用时分布，恢复后的片段不拼成完整成绩。
- 有参考/无参考分组目前来自冻结计划，实际 actor 输入、策略执行、视觉时效和自动重开尚未绑定，逐次明确标记缺口。`valid_complete` 仍是局部有效性结果，不能解释为冻结策略自主成功。
- 所有报告均禁止自动晋升；合成数据或诊断模型另有 `diagnostic_only` 标记。阈值被固定但尚未用于版本筛选。

后续软件工作包括实际执行/输入/时效凭据、有效决策率与平顺/控制权统计、停止与重开的批次执行接口，以及最终评估独立性登记。#4/#9 的真实自动重开和驾驶验收仍是完整票据要求；本切片与其后续软件测试均可在游戏关闭时完成。
