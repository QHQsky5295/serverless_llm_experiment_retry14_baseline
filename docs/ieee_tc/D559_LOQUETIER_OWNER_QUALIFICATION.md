# D559：Loquetier 所有权假设的第一关

2026-10-16。存储维护 D558 已结束并备份，随后在原生环境开展小型 CPU
对象生命周期检查，六项全部通过（含两项新增检查）。未改 serving、配置、
原生内核、工件或负载，也未运行新的 GPU 性能点。

实际虚拟模块、PEFT 和准备表示表明：纯推理准备后的权重与共享基座可在
原始 PEFT 对象回收后继续保持内容和指针一致；同时验证了共享存储情形，
不能将 Python 对象回收直接换算成显存节省。CPU 参考运算前后相等，不是
原生 SMLM GPU 输出或完整数值正确性的证明。

完整方法、历史与官方来源边界见 baseline 仓库
`docs/ieee_tc/D559_LOQUETIER_OWNER_QUALIFICATION.md`。
本仓库的小型证据及对应 baseline commit 位于
`paper_results/ieee_tc/loquetier/d559_owner_cpu/manifest.json`。

检查峰值578465792 B，high/max/OOM/kill/swap为0；真实进程树已释放。
下一步是复用现有原生容量入口验证真实 7B 最大分片的所有权、GPU 输出及
显存变化，再考虑安全并发。该 GPU 诊断尚未实现或执行，不重复本次六项
测试，不机械重跑旧低效配置。原始弱性能与失败证据继续保留。

本项回应“基线性能异常是否源自不合理适配”，不直接支持 PrimeLoRA 领先。
完整 7B W2/G2、PrimeLoRA 跨场景优化、3B、消融、机制、敏感性及论文文档
仍未完成。共享 Kubernetes 不变，HydraServe 既有隔离部署沿用。
