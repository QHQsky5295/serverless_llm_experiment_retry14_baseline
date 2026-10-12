# D472：CPU 准备候选的完整 W0 单轮验证

2026-10-12。前置 D471 原生 100 请求资格已通过。复用原 runner、原 4000 请求、
500 逻辑 adapter 工件池和真实远端只读交付，不新建 trace、权重或解压整池。

对照 D465，配置只新增已通过原生资格的
`resource_coordination.ieee_cpu_frontend_policy=fixed_uuid_cpu_frontend_v1`，
并使用独立输出/NVMe/HOST 路径。真实 model 配置、严格 service/preparation
profiles、SLO、九个公式、路由与 admission 语义全部保持；不回滚 D463，
不将其未获支持的性能假设包装为已接受。原 60 秒通知和实际物理计量不变。

可证伪假设与 D466–D469 相同：将无 GPU context 的前端初始化放到真实扩容
触发之前，能够减少触发后的启动等待与持卡时间，并改善完整回放的资源–服务
关系。CPU 准备仍占共同 CPU/内存，若其竞争导致退化也完整保留。不是缓存
旧物理状态、fork 已初始化 CUDA、GPU sleep 当免费资源或隐藏迟到等待。

本次新增执行键运行一次。先检验全部输入、原生 token、LoRA/物理绑定、
真实分配/释放与准备进程收尾，再按冻结 D394 共同阈值判断至少 3800/4000
联合达标，并报告真实 GPU-s/request。W0 预算只作本场景检查，不能替代
W1/W2/G2。不以短验证获胜保证完整获胜，不重复同键挑选更好结果。

同时核对自然 scale-down 后实际 physical return、CPU 替换准备和新激活；
未触发如实记录。启动/退出 CPU 开销、unused 准备和样本覆盖均保留。
原生后端/prepare policy 的研究依据见 D467–D469 文档及官方 vLLM v0.30
源码快照；不重新运行已否决的 pin-query、fork 或 import 微测。

执行前必须重新检查完整计划、内存/磁盘准入、无其他 GPU 作业与源码备份。
测量期间不执行分析、压缩、构建、推送或远端维护。结束后先实际释放，
再停止本轮远端服务、审计并封存数据。目标接近前不画图，不改论文。
7B W1/W2、HydraServe 独立资格、3B、消融与全部敏感性仍待后续顺序推进。
