# D400：ServerlessLLM 真实远程获取归属到服务内部

日期：2026-10-10。主线仍是已封存的 7B PrimeLoRA W0 → 7B ServerlessLLM →
其余 7B 比较及必要优化 → 3B → 消融、机制和敏感性。

## 本轮完成与边界

已在原有资格入口接入显式的 `published_gzip_ondemand_peft_v1`：请求实际进入
ServerlessLLM 后端后获取所需工件，复用共同严格 HTTP 下载器，不由回放器
下载。压缩对象来自用户已批准并完成的 D78/D80 发布缓存；生产缓存本轮没有
重新发布，也没有启动远程生产服务。不存在重新生成权重或执行负载。

这是披露的 PEFT 传输适配，不声称官方原版自动具备这一整条实验接入。
依据是历史 fair runner 已有按需远程模式，但将下载放在回放端；原生 controller
在注册期处理缺失工件时又会进入制品转换，而当前 vLLM 路径消费标准 PEFT。
本轮把普通按需传输移入服务域，仅跳过 LoRA 的 store 转换，保留 backbone
原生快速加载、RR、容量及扩缩容，不引入 PrimeLoRA 的缓存规划或调度。
其接入位置参考版本匹配的下载后构造 LoRARequest 模式。
[vLLM 0.10.2 文档](https://docs.vllm.ai/en/v0.10.2/features/lora.html)、
[ServerlessLLM 固定 controller 源码](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/controller.py)

| 检查 | 已获得证据 | 不能据此声称 |
|---|---|---|
| 同一工件并发获取 | 三个独立进程共享一次真实 HTTP 获取及发布回执 | GPU 模型路径已合格 |
| 不同工件并发 | 一个工件锁被占用时另一个仍完成 | 生产网络吞吐已测得 |
| 缓存与错误 | 内容/文件身份改变、无回执目录、404 均显式失败 | 可忽略失败或自动换本地池 |
| 取消 | 等锁及活动 I/O 的取消被保留，写入线程收尾 | 任意主机故障都可恢复 |
| 请求来源 | adapter、内容/清单 SHA、回执 SHA、transfer ID 关联 | 名称正确即可证明数值正确 |
| 时间 | 获取 span 位于后端进入至 engine dispatch 内，不重复相加 | 可从 TTFT 中扣除真实传输 |
| 回归检查 | 77 项 CPU 检查通过，7.600 秒 | 新的性能比较已经完成 |

测试使用临时微型字节夹具，不是重做实验工件。资源上限为 4 GiB、无 swap、
CPU 2/3/26/27。尚未运行新的 GPU 性能点，`remote_qualified=false` 保持明确。
每请求取消等待锁的检查间隔为 50 ms，作为适配实现开销披露，不把它当作
带宽或实验调参，也不从时延扣除。缓存只在本轮推理节点按需物化，不新建
永久完整工件池；命中要求已有内容验证回执和未变化的文件身份。

详细接口、测试和源码边界见 baseline 仓库
`docs/ieee_tc/D400_SERVERLESSLLM_PUBLISHED_RESOLVER.md`。

## 执行规范更新

最新用户要求已经追加到源计划，并保存完整新快照
`PLAN_APPROVED_20261010_FULL_NAMES.md`。后续汇报、文档和图表使用完整名称
PrimeLoRA 与 ServerlessLLM；历史原始 ID/路径、冻结指标 V1/V2 不重命名。
这只是展示规范改变，不重跑结果、不改变 95% 联合 SLO 或物理 GPU-s。

## 下一步与尚未完成

下一主线是补齐 ServerlessLLM 的真实 GPU 生命周期：包括 native store 的
持卡、活动/已退役引擎的进程身份，以及实际退出后的 release。复用现有
PhysicalGPUAllocation／NativeGPUCensus／部署记录和受限 launcher，不能拿
逻辑副本数或末尾 actor 查询代替完整物理持有。完成后只验证受影响的 7B
模型路径，再冻结合理配置，完整匹配回放一次。不重做 D72/D73 轮询对照。

推理机当前约 150.14 GiB，仍接近 150-GiB 启动底线；需重新核算请求触达的
冷缓存、最大临时文件、Ray spill 和日志增长后放行，不能借远端的独立门槛
放宽推理节点保护。原始结果与唯一工件不删除。

已有 7B PrimeLoRA 单轮只覆盖 W0，不代表 W1/W2、G2 或全部基线领先。
有效 Resident 预算参考、其余 7B 基线、3B、核心消融、机制与敏感性仍未完成。
CPU 检查不能关闭这些任务。当前不作新排名，也不为展示延误主线绘图。
