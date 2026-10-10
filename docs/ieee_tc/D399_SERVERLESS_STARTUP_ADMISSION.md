# D399：保留启动等待，不丢请求、不改业务起点

日期：2026-10-10。主线仍为已封存的 7B Prime W0 → 7B Serverless → 其他
7B 基线 → 对比后必要优化 → 3B。计划与冻结 V1／单轮 V2 均未修改。

## 本轮完成了什么

| 问题 | 本轮结果 | 证据边界 |
|---|---|---|
| 请求到达时原生 API/router 尚未建立 | 服务侧启动等待入口已接入现有资格入口 | CPU HTTP 验证通过，尚未 GPU 资格 |
| 等待是否被隐藏 | 原始到达／提交不平移；新增等待 span 是 dispatch 的子段 | TTFT/E2E 不减等待，也不重复相加 |
| 是否改变 Serverless 策略 | router 建立后交原生 RR/容量/扩缩容/加载路径 | 不等待引擎 warm，不加 Prime 策略 |
| 失败会不会被重试或删去 | 超时、取消、重复、原生错误显式记录；转发一次 | 不追改 D72/D73 失败记录 |
| 请求失败后是否仍取进程证据 | 实际 worker 读回移到清理前的 finally | 仅末尾读回，不冒充完整动态生命周期 |
| 回归检查 | 69 个 CPU 检查通过，2.822 秒 | 不属于模型性能实验 |

新增字段要求同一时钟域下满足：

`client_submit <= ingress_received <= ingress_forwarded <= native_http_received`。

发布端保持既有固定时刻的开放回放。等待入口位于服务资源域，不放在 4-GiB
外部回放域；先核验服务资源限制，再启动线程。正常接入只通过显式
`service_pre_ready_v1` 开关；旧诊断路径默认不变，新日志独占创建。

原生源码直接查找 router，缺少其创建前的排队入口，因此补齐部署生命周期
比延后业务起点或隐藏重试更符合冻结协议。启动期间由服务端缓冲请求可参考
Knative 的部署实践；这里不是复现 Knative，也不替换 Serverless 的调度和
自动扩缩容机制。[固定 Serverless 源码](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/app_lib.py)、
[Knative 请求流程](https://knative.dev/docs/serving/request-flow/)

## 对论文实验的意义

这一步解决的是公平计量与完整请求集合，不是证明 Prime 已经优于 Serverless。
历史“约四分钟等待”的轮询归因继续使用 D398 已核验的 D72/D73，不重复运行。
当前没有新 TTFT、SLO、GPU-s 或主表排名；也没有画图、启动远程服务、重新
发布缓存、修改 Prime、改论文或生成新工件／负载。

已封存的 Prime 7B W0 仍为 D371 的已知负载复用开发观测：4,000 请求、
联合达成率 97.70%、3.620631 物理 GPU-s/request。它不代表 W1/W2 或 G2 已完成。

## 尚须补齐，不能提前启动旧本地跑法

1. **真实远程交付。** 使用已批准并完成的 D78/D80 只读压缩缓存，服务内获取，
   不重打包、不重新生成权重或负载。原生 controller 在 router 建立前逐项
   注册 adapter；缺路径时 downloader 会进入 PEFT/store 格式制作，而当前
   vLLM 消费标准 PEFT 文件。不能只改成空路径，更不能悄悄改为 Prime 的
   按需驻留策略。下一步需选定并披露忠实的传输适配与准备时机。
2. **真实 GPU 生命周期。** 复用已有 allocation/census 组件覆盖 store、Ray
   和引擎持卡，释放必须有退出与原生上下文证据。修复后的末尾 worker 读回
   仍无法证明已缩掉副本的过去生命周期；缺少的 ID 继续显式拒绝，不放宽门槛。
3. **资源门槛。** 本机约 150.14 GiB 空闲，紧贴 150-GiB 底线；先估算冷缓存、
   spill、日志的新增峰值并核对可重建缓存，再允许新的重型运行。不删除唯一
   工件或原始测量。本轮 CPU 验证为 4 GiB、无 swap、CPU 2/3/26/27。
4. 完成上述缺口后，仅做受影响路径的最小 GPU 资格，再冻结公开合理配置、
   运行一次完整匹配 7B。其他 7B 基线、有效 Resident 预算、3B、消融、机制
   和敏感性全部仍在待办，不因为入口测试通过而关闭。

## 保护与备份

147 项历史保护核验和上轮一致：唯一与九月封条不同的是用户此前已修改的
`replay_openai_trace.py`；本轮 SHA 未变，未 stage。不能把该核验原始
`pass=false` 写成全通过。relayserve 用户修改、generated manifest、旧图和
原始结果同样未触碰。完整源码、测试和跨仓库版本由 D399 checkpoint 绑定。
