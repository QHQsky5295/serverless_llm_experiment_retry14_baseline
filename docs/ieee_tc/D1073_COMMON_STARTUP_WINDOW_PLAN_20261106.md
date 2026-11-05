# D1073：共同启动窗口诊断计划

日期：2026-11-06  
状态：运行前登记，诊断用途。

## 动机

D1071 和 D1072 使用相同的 120 s replay delay，但首个 PrimeLoRA runtime
初始化分别为约 50.6 s 和 116.7 s。当前 launcher 的 gate 只确认服务进程与
watchdog 已就绪，不确认首个物理 runtime 已达到可服务状态。因此 120 s 不能
作为稳定的启动屏障，D1072 的前期排队会污染后续联合 TTFT/TPOT。

## 设计

使用一个新的、预登记的固定共同窗口：`FAASLORA_TC_REPLAY_DELAY_S=240`。
PrimeLoRA 与 ServerlessLLM 后续匹配运行均使用同一窗口、同一 trace 起点
规则和同一生命周期计量；等待期间 GPU 持有时间计入 U，不从成本中扣除。
本轮仍不把“ready 后平移 trace”当作合法替代，不按系统启动快慢分别选择
窗口。240 s 只是覆盖已观察到的约 117 s 初始化方差的保守诊断上界；若
两系统均需更长窗口，另行预登记，不能事后按结果调整。

## 固定项

- 3B W1、4,000 请求、seed 42、500-adapter subset、配置和 generation
  contract 与 D1071/D1072 完全相同；
- local-sim no-delay、`prepublished_gzip_v1`、request-time pack=0；
- D648 共同 TTFT/TPOT 阈值与 3,800/4,000 联合门槛不变；
- 不改代码、公式、adapter pool、trace、SLO 或 baseline。

## 判定

先检查初始化/ready 到 replay 起点的余量、首批请求的 dispatch wait、服务
TTFT/TPOT 和 GPU 生命周期；再按 D648 重新计算联合达成。D1073 不是正式
留出结果，也不替换 D1071/D1072。只有在共同窗口下 PrimeLoRA 与基线均有
可比完整点后，才进入正式候选冻结和后续机制消融。
