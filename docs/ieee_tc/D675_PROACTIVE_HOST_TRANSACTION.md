# D675：主动 HOST 激活的事务内存储证据复用

2026-10-18。这是 D674 选定的唯一候选实现和 CPU/原生方法绑定资格，
不是新的 GPU 性能结果。D672 的3B完整结果仍为联合3647/4000（91.175%）、
14931.314879785059 GPU-s，仍差153条达标；数值正确性证明仍未知，G1/G2未闭口。

## 证据、假设和实现边界

D674复用D668记录发现982次主动准备具有三次HOST清点，其中127次在前两次
之间改变范围/数量/字节。不能复用最早的replacement-cost-provider图；相同
计数本身也不能证明对象不可变。D329反例、D352 GPU-hit、D663受保护请求
激活、D671独立图复制的历史边界均保留。本轮查看了上述修改历史和当前调用链。

本轮再次核对官方[vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
及[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)：
已注册、non-inplace激活读取现有CPU对象，向GPU槽复制；它不同于冷加载、
CPU注册和移除。官方代码支持这条执行路径的分析，不提供本机性能结论。

候选仅省略**本次激活的预算后检查**中的重复存储遍历。先完成原来的候选
评估、HOST victim removal/staged registration及GPU victim处理，随后仍
取得新的完整预算前图。新的私有回调只由主动路径传递，同时携带当前HOST
source incarnation。实际worker验证：

- 调用的是绑定在该owner上的精确pitched-copy闭包，而非任意替代loader；
- 原生类型、版本、方法绑定、dense/TP=1及无需转换的复制合同成立；
- 目标已注册、不在staging，前后成员/对象/模块及源incarnation一致；
- 没有owner yield、CPU注册/移除/packing跨越这一复用区间。

这不是epoch/TTL缓存，也不是一个可由路由器设置的“跳过检查”开关。普通
请求不能借用主动回调；即使两个fixture loader是同一函数也不获得该资格。
不同事务重新遍历。缺少合格回调或recheck时仍完整检查；资格验证抛错不被
吞掉；实际复制、fence、pin/release及GPU victim/loss决策不变。

后检查仍新读allocator并执行workspace/预算判断；allocator增长可拒绝，
复制或后置身份异常仍使owner失效。新的回执标识为
`same_transaction_proactive_host_activation_v1`，与D663请求路径和GPU-hit分别计数。
九公式、实际准入判断、路由、profile、SLO、负载、模型配置和远端交付均未改变。

## 验证及保留的失败

复用原有测试、D663有界资格脚本及D668精确身份关闭入口，不搭建新实验框架。

| 阶段 | 实际结果 | 解释 |
|---|---|---|
| qualification | 695项，12 errors | 3项旧环境不存在vLLM0.30模块；9项新fixture误带计划ID却没有objective，在复制前被正确拒绝 |
| analysis | 692项全部通过 | 修正fixture计划调用，旧CPU环境回归；不包含3项原生绑定检查 |
| qualification2 | 33项，1 error | 原生绑定及其余32项通过；跨模块复用fixture时遗漏其inline-planner setup，真实子进程启动超过fixture的5秒限制 |
| analysis2 | 2项，1 error | 复核同一setup遗漏；旧环境也在planner启动处超时，不是HOST激活失败 |
| qualification3 | 33项全部通过 | 复用原模块完整setup/cleanup，在实际vLLM0.30环境验证；不提高测试超时、不改生产planner |

后两个完整事务fixture覆盖冻结目标/非LRU victim及HOST fallback，以及先
staged注册/替换再进行新鲜预算前检查。其测试不声称测过真实CUDA复制或
GPU数值输出。既有actual-worker检查同时覆盖新回调的loader绑定，以及
workspace、allocator变化和原有D663/D352路径。失败与修正后的各日志均保留。

五个资源域已按InvocationID/inode核对递归空并关闭，峰值分别为
615886848、543141888、3209060352、1670012928、659804160字节；所有
high/max/OOM/kill/swap均0，无GPU作业。较大峰值来自错误fixture触发的
planner子进程，已退出；不把它包装成服务性能测量。147旧保护项和3份用户
修改SHA均未变化。没有重建数据、删除证据、启动远端服务或触碰非项目k3s。

生产SHA：

- gpu_monitor.py：`7ed440e00431f03af907155a730b0075e5bf4919c9a3203260bae4cfb3cc6def`
- residency_manager.py：`0cacfa5fc923e0c911ff1a62a0c26fdc9f5f7830e4eb4687edcec81810bd4f17`
- 最终测试文件：`5eb727658f12aa4e8a8171344cacda045af8138090991872ed4f173f6803af62`

原始证据：`results/ieee_tc/p2_backend_qualification/d675_20261018/`。
curated：`paper_results/ieee_tc/p2_backend/d675_proactive_host_transaction/`。
按github-sync执行显式路径、diff、来源/旧保护、secrets和测试检查后备份；
推送是否成功以backup_receipt为准，不将待发布写成已发布。

## 下一主线

先保存这一个可回退候选。下一复用D664/D659的100请求原生GPU smoke，
保留旧标识并检查新增主动回执实际触发、源和预算不变量；通过后才能执行
一次与D672同配置的独立4000请求W0。请求中不打包、单键不重复寻优、
不开详细profiling、不改评价尺子。不能把减少一次调用直接写成TTFT收益。

3B W1/W2、3B完整baseline/dLoRA、同版本7B闭环、A1–A5、S1–S13、
成本/CE、两轮审稿与最终图表全部保留；达标前不画图。HydraServe已完成的
必要隔离适配不重做，非项目k3s保持不变。本轮技能指导了有界验证、失败
留存和明确路径备份，没有采用技能中的通用吞吐目标替代冻结协议。
