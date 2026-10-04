# Δt BC：CPU 线程数与重载断言

2026-10-02。主分支 `5e90bcd` 全量测试结果为 **1147 passed, 1 failed / 4928.76 秒**。
失败位于 `test_temporal_model_uses_same_features_in_live_numeric_seam_and_exact_replay[fixed]`：
训练时预测与重载推理相差 `7.450580596923828e-9`，原断言要求逐位相同。
本地证据：`runs/t08-native-entry-integrated-full-20261002-results.xml`。

保留该次实际权重和数值输入，通过公开 `NumericInfer` 分别使用 1/2/4/8 个 CPU 线程。
四次输入特征完全相同；1 线程复现原差异，2/4/8 线程误差为零。
训练固定为 2 线程，结束后恢复调用方设置，因此训练与后续推理可能使用不同计算内核。
诊断结果保存在 `runs/t08-temporal-roundoff-probe-20261002/comparison.json`。

修正仅涉及测试：跨训练/重载比较采用已有的绝对误差 `1e-6`，相对误差为零；
不改变运行时线程数、输入特征、权重或控制逻辑。同配置下的数值精确重放与固定时间特征等值检查保持不变。
新增 1/2 线程与 actual/fixed 两种模式，共四种组合。修改断言前 fixed/1 线程复现失败；
修改后 **4 passed / 2.77 秒**。新的全量回归尚待执行，不能用局部结果覆盖原全量失败。
