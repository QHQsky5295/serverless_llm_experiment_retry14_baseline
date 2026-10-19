# D697：3B 完整 W0 单飞扩容候选（预登记）

日期：2026-10-19  
状态：启动前登记，尚无性能结论

## 科学问题

D695 的完整 W0 观测显示，同一控制评估可能在同一请求前沿同时提交多个
activation，随后就绪顺序改变初始副本承载分布，并伴随 admission→engine 和
native decode 退化。D697 只检验一个可证伪变化：在 `ieee_confirmed` 路由下，
每个控制评估至多提交一个物理 activation；前一 activation 登记 ready 后，后续
控制评估才可继续提交。该变化不修改论文扩缩容公式、预算、handoff 或 SLO。

## 冻结合同

- 模型：Llama-3.2-3B-Instruct；4 张 RTX 3090，TP=1。
- 负载：既有 4,000 请求 W0 trace、500-adapter 子集、seed 42；不生成新负载。
- 生成：`fixed_length_greedy_v1`，原生 vLLM token-id 计数，最多 256 输出 token。
- 资源、远端真实交付、GPU 生命周期、共同阈值和分析口径与 D695 逐字段一致。
- 代码：生产候选含已提交的 `325fe9822c694eee277f6c2d0fac9101e93cd9a4`；
  其余生产回退基线为 D693 父版本。
- 配置：`configs/ieee_tc/20261019_d697_3b_full_w0.yaml`；入口脚本为既有
  `run_d293_7b_prefetch_validation_100.sh d697`。
- 所有输出使用独占 D697 路径；若路径已存在，启动器必须拒绝覆盖。

## 预登记判定

必须保留 4,000 请求计划分母和全部失败。完整运行只有在正确性、原生生成、
adapter 绑定、请求阶段计时、物理 GPU 分配/释放及远端交付证据均完整时，才可
进入候选比较。候选至少需要实际出现可审计的 scale-up submission/ready 序列；
若未触发，标记 `qualification_only`，不据此宣称性能改善。

若触发并完成全量回放：联合 SLO、P95/平均 TTFT、TPOT、E2E、生命周期 GPU-s
和阶段分解与 D682/D695 并列比较。只有联合 SLO 或目标资源/尾延迟方向改善且
无新的正确性/资源回归时，候选才可进入下一轮开发冻结；否则回退到 D693 父版本。
无论结果如何，D697 原始日志和分析保留，不重复同一执行键寻找更好结果，不画
论文图。

## 启动前资源与远端门禁

启动时重新检查 GPU 空闲、主机内存公式、每个文件系统的 150 GiB 启动线与
32 GiB 预测增长、CPU/cgroup 归属、owned 远端 3B 服务健康和 1-Gbps/full 链路。
若任何门禁不通过，本执行身份保持 `not_run_safety`，不伪造结果。运行结束后
按既有 close-launch → 受限停止远端 → close-management → 受限 native 分析 →
封存顺序收尾。
