# D539：ServerlessLLM W1 等待与启动证据核查

2026-10-15；仅离线复用 D537 W1 和 D414 W0，不新增性能运行，不改服务代码、
配置或历史结果。遵循完整 PLAN、冻结指标 V1 与单轮 V2；不作运行间统计推断。
分析使用 analyze-results；封存使用 github-sync 的明确路径和安全检查。

## 观测结果

两轮各4000请求，输入序列相同但到达时间不同，因此下表不是仅改变启动的
受控因果对照。条件分组由系统自身行为产生，不能作为随机对照。

| 指标 | D414 W0 | D537 W1 |
|---|---:|---:|
| 平均用户 TTFT (ms) | 973.687 | 20531.568 |
| P95 用户 TTFT (ms) | 1052.246 | 109994.301 |
| 平均 router queue (ms) | 522.210 | 19873.307 |
| P95 router queue (ms) | 0.973 | 109382.314 |
| 平均 ingress wait (ms) | 1.615 | 179.867 |
| 平均 adapter resolve (ms) | 13.940 | 33.254 |
| 有请求的后端 ready 实例身份数 | 5 | 16 |
| 首个后端 ready 相对业务开始 (s) | 77.483 | 180.412 |

W1 平均 router queue 占平均用户 TTFT 的96.794%；这是同请求集合的均值之比，
不是每请求比例的均值，也不是可删除的延迟。D537 的平均 service TTFT 为
462.918 ms；adapter resolve 的具体边界与其他 span 可能重叠，不重复相加。

W1 入队时无 ready 实例的282请求，平均 router queue 为123189.895 ms；
有4实例的2638请求，平均仍为12569.897 ms。因已有积压和容量限制，
“已有 ready 实例”不等于“可以立即接纳当前请求”，也不能据此断言GPU算力饱和。
八个阶段的平均 queue 分别为109620.827、11287.613、5255.602、12769.147、
9246.997、7745.295、2728.405、332.569 ms；前500请求联合SLO为0/500。
完整4000请求始终保留，不能把第一阶段删除后称为主结果。

## 启动证据与官方实现

逐一绑定16份 native ready receipt 的 PID、SHA、时钟域及独立 worker日志，
16/16都出现已编译模型图的缓存命中。最早四实例：

- ready 相对业务开始180.412–180.570 s（部署通知再早60 s）；
- router 实例创建至 ready 的同主机 wall-clock 差130.962–134.077 s；
- 模型加载日志3.163–5.068 s；Dynamo转换25.91–27.58 s；
- 编译图缓存读取7.155–7.180 s；引擎profile/KV/warmup42.30–44.54 s。

后续12实例的创建至 ready 为66.909–75.004 s，Dynamo转换10.51–11.73 s，
引擎profile/KV/warmup20.87–22.80 s。这些日志区间有嵌套，不能求和当总启动时间；
native单调时钟和wall-clock亦未混合相减。16是实例身份数，不是16张GPU，
物理资源仍按D537实际四卡store池持有区间计量。

vLLM 0.10.2 将Dynamo追踪/转换与Inductor产物缓存区分，因此命中编译图仍可能
有初始化开销。观测排除了“这些模型图全部重新编译”的解释，不排除其他
kernel首次准备、进程初始化或资源竞争；初始并发启动更慢只形成线索，尚无
单因素证据证明CPU争用为唯一原因。
[对应版本官方文档](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/docs/design/torch_compile.md)

D537 与D414执行router文件SHA一致。相对官方commit
`9f50241baa5386e06a9321c51f19a9ef5f964c2b`，
四个函数 auto_scaler、_auto_scaler_loop、_stop_instance、_finish_instance
的AST比较全部一致。冻结min1/max4/target4/keepalive10不变；W1相邻阶段30秒
间隔与自然缩容/重建均保留。只移除历史无条件ready轮询的既有修复，不更换
round-robin排序、满容量等待或缩容策略。
[官方router](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)

本次没有找到足以支持再改baseline的必要适配错误；这不是对所有未观测路径
的无缺陷证明。原论文快速权重加载与本轮多租户排队/完整进程启动的边界不同，
不能把用户TTFT当作原论文checkpoint加载速度的反例。
[ServerlessLLM论文入口](https://www.usenix.org/conference/osdi24/presentation/fu)

## 裁决与主线

1. D537保留为该适配版本/冻结配置的单轮W1观测：44.275%联合SLO，不达95%；
   不以较差结果为理由改参数或重复相同执行键。
2. 历史“四分钟等待”已有轮询修复证据；本次W1等待不再用旧成因一概解释，
   不把所有延迟归为远端传输、重编译或ServerlessLLM架构固有缺陷。
3. 原生绑定/生成/计量与独立数值正确性分别记录，后者仍未知；不补写n_correct。
4. 返回7B W1主线，优先复用S-LoRA资格及脚本取得W1点，再补Loquetier和
   HydraServe；不重做已封存W0，不扩展无关基线调优。
5. W1/W2/G2、3B、A1–A5、S1–S13、成本与审稿文档仍未完成。PrimeLoRA
   D533的93.725%联合SLO也未达95%，不能以胜过当前两条W1基线宣布目标完成。

证据位于`paper_results/ieee_tc/main_comparison/d539_serverlessllm_queue_review/`。
RAW为原项目`results/ieee_tc/serverless_qualification/d539_20261015`。
审计资源域实际关闭，峰值513466368字节，high/max/OOM/swap为0。
