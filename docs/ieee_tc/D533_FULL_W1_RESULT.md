# D533：PrimeLoRA Full 在 W1 突发负载下的单轮结果

2026-10-15。沿用 D476 的策略、参数和初始化 profile；只切换既定 W1 到达视图、
独占结果/缓存路径及 D534 共同编译工具链。D529 元数据候选关闭。
执行源 fc4161a1088efe1b49d853dbbc8effe9a525c431；baseline 源
bde215460acb9e850efa479a89bda6fc39a2bd7d。无同键重跑，无新权重或负载。

## 完整观测，不宣称达标

| 指标 | PrimeLoRA D533 | vLLM D536 |
|---|---:|---:|
| 完成且原生输入/adapter/数量绑定合格 | 4000/4000 | 4000/4000 |
| 实际输出 token | 458224 | 458224 |
| 联合 TTFT/TPOT 达成率 | 93.725% | 72.675% |
| TTFT 达标数 | 3788 | 2907 |
| TPOT 达标数 | 3950 | 4000 |
| 平均 TTFT（ms） | 1625.270695 | 2900.834480 |
| P95 TTFT（ms，Type-1） | 4473.332113 | 9777.026766 |
| 平均 TPOT（ms） | 35.244626 | 27.169028 |
| P95 TPOT（ms，Type-1） | 58.251854 | 31.402243 |
| 平均 E2E（ms） | 5649.552374 | 5976.167070 |
| 生命周期 GPU-s | 8145.351609 | 9008.761400 |
| GPU-s/原生绑定合格请求 | 2.036338 | 2.252190 |

两系统逐请求 source、adapter、target、canonical/native prompt SHA 相同，源 trace SHA
efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c，W1 view SHA
103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3。
source seed42/development41 身份不改为独立留出测试。只有一轮，无运行间 CI。

PrimeLoRA 观测 GPU-s 低9.5841%、P95 TTFT 低54.2465%、联合达成高21.05个百分点；
TPOT较高也必须保留。**3749<3800：PrimeLoRA 尚不满足共同95%要求。**
不能把“比当前 vLLM 点好”写成达标、所有 baseline 最优或跨场景最优。
G2参考尚未正式冻结；启动前记录的 D536 候选 B=6756.571050 GPU-s，
PrimeLoRA U/B=1.205545，连这一候选预算也未满足。不能用 W0 的较大预算替换。
独立完整模型的数值正确性仍未证明，n_correct 保持 null；上表分母明确为原生绑定
合格请求，不将 HTTP 成功或正确 token 数冒称独立数值证明。

## 阶段及机制证据

八个预定义500请求阶段的联合达标数：409、387、499、488、489、491、491、495。
前两阶段合计204次违约，占全程251次违约的81.27%；其余47次仍保留。
对应 P95 TTFT（ms）：27800.744、7530.022、1834.672、2155.109、2438.986、
2379.657、2500.141、2354.413。不能删除前两段重算主结果，也不能从这组相关性
直接断言全部损失来自编译、扩容或远端。

实际生命周期：initial 1次、natural scale-out 5次，均 runtime-ready；不是8次扩容。
initial trigger→engine-ready 90.499551s。物理占用四个互斥窗口：
pre-arrival35.463717、arrival8076.458608、drain3.670815、cleanup29.758469 GPU-s，
和为8145.351609。CPU-only准备不虚构GPU持有。
dispatch-time层级4000条完整：GPU2736、HOST321、NVMe809、remote134。
真实fetch132次/131480060线上字节；请求中打包为0，服务端与客户端逐transfer吻合。
dispatch remote 数不等于fetch数，不能混为命中指标；数量相同也不能证明缓存无效。

原生TTFT/TPOT重算、完整E2E阶段恒等式均通过1ms门槛。均值阶段可加，
P95与嵌套span不可加。详细stage_means_ms、全部阶段与触发计数见audit.json。

## 编译初态与证据限制

两系统实际入口使用相同 D534 编译环境与 D535 通用采样软件；实际worker回执核对一致。
PrimeLoRA initial 模型图使用新prefix3a5577e355，运行前直接metadata清单未包含该prefix；
实际日志显示graph编译8.81s、总torch.compile30.50s、engine内部初始化56.49s。
后五个激活worker直接加载本次生成的AOT图，编译0.27–1.81s，内部初始化13.05–14.52s；
另一个CPU-only候选被abandoned。vLLM D536首次图编译9.01s、总31.58s、初始化56.71s。
两者启动和第一次LoRA调用均保留实际JIT时间，不事后扣除或补预热。

这排除了“PrimeLoRA initial直接命中D536完整模型图”的解释；不能证明全部低层kernel
缓存初态一致。两者原生LoRA配置仍不同（PrimeLoRA14slot/rank16，vLLM4slot/rank64）。
因此只报告完整配置的单轮观测，暂不冻结严格初态匹配的W1 G2参考或作单机制因果归因。
已完成的首次输入核对曾把baseline回执误找在日志文本中，实际存在worker_receipts JSON；
失败脚本与回执保留，修正仅涉及离线证据位置，没有重跑推理。缓存目录正则的过度转义
导致中间JSON该投影为空；最终curation从已保存原始日志行严格恢复，原中间JSON不覆盖。

## 安全、保存和下一步

服务内存峰值20661305344B，high/max/OOM/kill/swap全部0；全部GPU实际释放。
本地service/launcher scope及分析scope精确核验为空并关闭；远端仅本项目两个工件服务和
监控被停止，InvocationID逐一核验，端口无残留。共享k3s从未改动。
147个旧投稿保护文件、用户dirty文件保持不变；无图表生成或论文源文件改动。

原始结果：results/ieee_tc/p2_backend_qualification/d533_20261015（original项目）。
curated：paper_results/ieee_tc/main_comparison/d533_full_w1/result1。
下一主线：保留本轮未达标点，继续既定W1公开baseline（ServerlessLLM优先）及W2共同输入，
结合已保留启动/排队证据确定一个可证伪的PrimeLoRA优化方向；不得退回无止境W0局部调参。
任何新改动保持九公式、跨场景同配置、全部历史与共同评价尺子，不能靠重复现有键取得胜出。
3B、G2、dLoRA B6、A1–A5、S1–S13、成本/CE、审稿文档和最终图表仍未完成。
