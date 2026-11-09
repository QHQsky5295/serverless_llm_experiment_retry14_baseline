# D1172：无效入口尝试的协议裁决

日期：2026-11-09。该目录对应 D1172 候选的第一次启动尝试，**不属于性能
结果，也不进入任何 PrimeLoRA 或基线比较**。

## 观测

- 需求加权选择器在回放前选出 459/500 个 adapter，部署期物化 wall time
  约 246.14 s，传输日志和 NVMe 文件均保留。
- 复用的通用 `run_d1100_7b_busy_service_rate_w2.sh` 没有继承 D947 barrier
  入口中的 `FAASLORA_TC_REPLAY_DELAY_S=420`。
- 因而 replay publisher 在部署物化完成前释放了到达；service.log 显示首批
  请求在启动阶段排队，早期 `dispatch_wait` 约 295 s、SLO 暂为 0%。
- 该运行随后被主动停止；GPU 已回到 15 MiB/卡、远端 invocation 已 inactive，
  `launch.json` 为 `pass=false`。

## 裁决

这是部署—回放 barrier 缺失导致的协议失败，与需求加权选择器的性能无关。它
不能被当作“预物化降低/恶化 PrimeLoRA 性能”的证据，也不能覆盖或替换 D947、
D1118 或任何历史结果。新的有效尝试必须使用全新的执行键、结果目录和缓存目录，
并显式保留 D947 的 replay delay；本目录作为失败证据永久保留。
