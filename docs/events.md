# 赛事生命周期（T03，开发中）

入口为 `run_experiment(EventRun(config_file, output_dir), event_environment=...)`；仍使用同一实验运行与回放边界。当前提供状态机、证据保存和合成环境测试，**尚无 Windows 实机适配器及自动重开 CLI**。不要将 Computer Use 的逐步菜单验证计为程序无人值守能力。

## 状态与证据

运行依次记录准备、起跑核验、尝试结束、重开和恢复核验。每次起跑都有独立 `attempt_id`。重开必须返回匹配车型/PI、低于 3 km/h、位于指定起点半径内的活动驾驶画面，才创建下一次尝试。超时结束记为失败；稳定的完成页面记为 `completion_observed`，仍不是经过路线、倒带和接管检查的有效成绩。

`EventEnvironment` 注入遥测、带单调时间的灰度画面、前台状态和停止信号；`pulse` 只支持有界的菜单按钮脉冲，返回前必须释放。外部适配器负责限制 `read/pulse/close` 的实际耗时及独立停止保护；状态机无法抢占卡死的驱动调用。当前没有驾驶动作输出。

画面必须尺寸正确、时间新鲜且递增。多个图块同时匹配才识别一个页面；同时匹配多个页面按未知处理。菜单动作要求连续两张匹配画面，每一步只发送一次；未出现下一状态则在阶段时限内退出。用户停止、失焦、断流、画面异常、运行中的车型变化或时间/位置跳变均中止。`IsRaceOn` 变化本身不算完成。

## 配置与回放

配置沿用版本 1 的车辆/调校/辅助快照，增加 `event_run`：

- `version`、`conditions_verified`、本地 `verification_evidence` 路径。
- `max_attempts`（1–10）、准备/重开时限（0.5–120 秒）、单次尝试时限（0.5–1800 秒）。
- `expected_car_ordinal`、`expected_pi`、`start_position_m` 和 `start_radius_m`。
- `screen_size`、`signatures`：每个页面包含 `box: [left, top, right, bottom]`、`template` 和 `max_error` 图块。
- `start_steps/restart_steps/finish_steps`：按顺序指定期待的 `screen` 与要发送的 `button`。

模板采用 `P5\n宽 高\n255\n` 后接灰度像素的 PGM。配置和证据在观察前固定并复制到本轮目录，原文件后续变化不影响本轮识别。条件未核验时只保存观测并退出，不发送菜单动作。完整配置示例见 `tests/test_event_run.py` 的合成夹具，不能用于真实游戏。

除原始遥测外，保存 `event-config.json`、`event-assets/`、`frames/`、逐条刷盘的 `event-journal.jsonl` 与最终 `event-run.json`。回放核对原始遥测/画面/资产哈希、日志与最终摘要；缺失或损坏时从可读事件前缀恢复尝试列表，报告证据不完整，不补造释放成功。哈希用于发现损坏和不一致，不是对整个目录的防篡改签名。逐帧记录会占用较多磁盘，实机适配器须选择足以识别的低分辨率并测量吞吐。

`fh5 replay <run> --report <new-report.html>` 可离线查看尝试和菜单动作。`unattended_verified` 始终为 false；真实条件和剩余验收项见 [T03 验证记录](validation/t03-event.md)。
