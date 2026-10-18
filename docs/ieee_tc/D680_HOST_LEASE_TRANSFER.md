# D680：同一执行者内的 HOST→GPU 引用交接

2026-10-18。当前为唯一候选的实现及正确性验证；尚无新性能结果。
D677完整3B联合90.700%、U=15894.960797 GPU-s，G1/G2仍未达标。
独立数值资格未知不改成通过。PLAN、冻结V1/单轮V2及两模型完整范围不变。

## 历史、原始来源与可证伪假设

D678已有1438条受保护HOST路径：最终native load均值15.475ms，
acquired→post-snapshot 1.320ms，post-snapshot→dispatch 142.000ms，
其中独立HOST release RPC wait均值68.907ms。这些区间重叠，不能相加、
从旧TTFT扣减或预言172条联合缺口可全部弥补。新假设是：GPU引用已经
确认保护之后，仍等待一个独立源引用释放往返，会增加请求开始等待，也向
正在decode的执行者增加控制消息。去掉该往返可能改善服务质量；须完整测量。

核对D333、D355、D662、D663、D675及D677/D678历史：pending确认与HOST
保护之间有控制器可观察更新，不能合并；D355原子控制交错已经存在，不重做；
不再优化已降至约1.3ms的局部后快照清点。D595/D628异步/并行退化仍是反证。

本次在线核对官方[vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)、
[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
及[engine core](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)。
native LRU操作依赖单线程core的串行调用；CPU注册对象与GPU激活/保护应区分。
[PyTorch非阻塞拷贝文档](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
说明pinned HOST源在拷贝结束前修改可能破坏GPU数据。这里保留实际copy/fence，
不通过异步释放或省略fence减少等待。来源不提供本机性能保证。

## 唯一候选及安全合同

显式候选开关 `model.ieee_host_lease_transfer=same_owner_host_to_gpu_v1`；
未开启的历史配置不改变。它不是失败后回退或新调参阈值。仅对已持有明确
HOST demand-preparation lease的请求，在同一个native load/acquire事务中：

1. 检查相同owner、adapter、路径、源incarnation和确实持有的HOST lease。
2. 保留所有实时容量检查、原生加载、copy completion fence与GPU/CPU pins。
3. 成功取得GPU lease并完成预算后检查，才释放该请求的HOST lease。
4. 返回明确的源/GPU lease、获取/释放epoch和时间；post-load identity包含
   释放后的owner epoch。控制器严格验收后结清HOST所有权，不再发送释放RPC。

其他读者、原有borrowed pins和GPU执行引用保持；最后GPU release才按原规则
解除其保护。源变为同内容GPU时仍用原有冲突重观察，不改dispatch-time HOST
分类。拒绝/容量等待不消费HOST lease；拷贝或后检查异常仍使owner失效。
重试绑定同一交接ID，不能换lease重用结果；丢失/损坏确认继续保留未决请求，
不生成、不推定容量已经释放。native pin并非CUDA host allocator的pin计量。

九公式、路由/冷热需求、主动admission、pending注册顺序、profile、共同SLO、
扩缩容阈值、生成契约、远端缓存/真实传输和资源包络均不改变。

## 验证与下一主线

复用D663受限测试入口与D668精确空组关闭；测试添加在既有三个模块，不新建
实验框架。覆盖正常交接、重试、其他读者、未知源、失败保留、并发GPU promotion、
丢失/损坏确认及core消息合同。实际结果：

| 阶段 | 结果 | 范围 |
|---|---|---|
| qualification | 526项通过，4.560s | 旧合格CPU环境的新增交接和既有生命周期回归 |
| analysis | 1项失败，证据保留 | 新增worker重复确认测试揭示旧缓存回执漏公共身份字段 |
| qualification2 | 72项通过，2.158s | 实际vLLM 0.30环境的方法绑定、交接、控制器和重复确认 |
| analysis2 | 527项通过，5.093s | 修正后的完整受影响CPU回归 |

后三组与前组有重叠，不相加当作独立实验或统计样本。失败的原因是原生
`admit_and_demand_load`缓存命中直接返回内部事务字典，绕过公共clock/worker/
allocator-policy字段封装；首次回包有字段，重放缺字段。将两条路径统一至
同一确认封装，不重复加载、fence或释放，也不伪造新获取时间。此修正是同一
交接协议所需的重放正确性修复，不据此推断旧性能运行受影响。

四个辅助资源域按InvocationID/inode核对递归空后关闭；峰值分别593850368、
590106624、702812160、583024640字节，high/max/OOM/kill/swap均0。
147旧保护文件及三份用户修改SHA不变，PLAN/V1/V2不变。准确代码版本和
失败前/后源码按测试回执SHA核验保留，备份按明确路径执行，不stage用户manifest。
第一次发布在解析保留的单项失败测试计数时中止：旧工具只接受`tests`，而日志
为`1 test`；没有commit/push。修正单复数解析后复核已生成源码归档，在新的
`paper_results/ieee_tc/p2_backend/d680_host_lease_transfer/qualified/`发布；
原中间归档保留，不重跑测试或性能实验。是否推送成功以publication回执为准。

无GPU、无远端启动；不修改非项目k3s、不生成权重或负载、不覆盖旧结果。

只有正确性和原生方法资格通过后，复用既有100请求真实3B资格；新增交接必须
实际触发并符合引用/epoch/时间不变量，然后新版本一次完整W0。不得仅凭少一次
RPC宣布收益，不能跳过完整回放。3B后续、同版本7B、baseline/dLoRA、消融、
机制/敏感性、成本/CE和最终论文材料全部保留；接近目标前不画图。
