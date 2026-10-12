# D471：原生 CPU 准备路径的原 100 请求验证

2026-10-12，执行源码 `862ac54bf1eabd9a1a61274a033c4653f64ce1b4`。
这是正确性与计量资格，不是完整 W0 或最终 G1/G2 结果。

| 项目 | 单轮观测 |
|---|---:|
| 请求/原生输出与内容绑定 | 100/100 |
| 原生输出 tokens | 17369 |
| 共同联合 SLO | 100/100 |
| 平均/P95 TTFT | 708.181 / 1823.181 ms |
| 平均/P95 TPOT | 30.209 / 41.174 ms |
| 物理生命周期占用 | 374.566427 GPU-s |
| initial / natural-scaleout ready | 1 / 1 |
| 真实远端 transfers / wire bytes | 29 / 50650865 |
| 请求内打包 | 0 |

原生计时重算误差为 0 ms。4 个固定 UUID 的 CPU 前端全部在共同部署通知后
建立，均位于原服务资源域；2 个按原 factory 获得物理分配并推理，2 个未启用
且收到关闭后正常退出。准备与授予权限时的上下文检查均为未初始化 CUDA、
无 current/primary context。相应权限前的 GPU census 样本数为 12/107/250/250，
均未观察到该设备上的服务 context；离散采样不声称排除所有瞬态。

两个真实 journal 均为 acquire → prepared-worker-bind → native-workers →
native-workers-exited → release。首次 acquire 比 CPU-ready 早约 0.733 秒，
该等待完整计入持卡时间；不能先等准备完成再推迟记账。两区间之和与原积分
374.5664273959119 GPU-s 相同。CPU-only 准备不虚构 GPU 时间，但其 CPU/内存
仍在共同资源包络中。短回放没有中途缩容及替换补充，不额外重跑凑触发次数。

服务峰值 12906348544 B；262 个监控样本，high/max/OOM/swap 均为零。
服务及 watchdog 退出码均为 0，实际 context 清空及服务路径移除后，才停止
本轮捕获身份的远端服务/监控并取回日志。远端链路为 1000 Mb/s/full。
所有请求的 dispatch tier 完整，HOST 准备 27 项（residency 23，handoff 4）。
退出期 engine_dead 命名的保留目录仍保留；不根据目录名断言推理失败，实际
请求无失败、worker returncode 0、物理 release 及测量完整性均通过。

源结果 SHA `bbdd9be54205396712022edf25ab8a9aafc94c9d761f855f8d8e91f7aadfa88b`。
原始独立数值正确性标记仍为 null；内容绑定不冒称独立 kernel 数值证明。
147 项历史结果保护通过。公开派生数据位于 `paper_results/ieee_tc/p2_backend/d471_native_preparation/`。

结论：允许进入一次新键的原 4000 W0 验证；不从这段短负载外推资源节省，
不替代 D465 非达标结果，也不宣布跨场景最优。D463 仍是父实现的一部分，
其性能贡献未被接受。封存基线不优化、不重跑。
