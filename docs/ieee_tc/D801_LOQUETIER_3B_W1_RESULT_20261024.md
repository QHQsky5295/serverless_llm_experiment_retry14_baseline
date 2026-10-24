# D801：Loquetier 3B完整W1单轮结果

## 本轮完成了什么

沿D799/D800的同一原生配置完成W1：相同4000请求序列分为8×500 phases，
phase内到达间隔减半、phase间30秒。原生SMLM、四个125-ID静态分片、batch4、
顺序初始化、保留PEFT owners和D797模型内核全部不变。静态分片不是弹性
控制器；W1是共同的突发/扩容压力场景，不声称Loquetier发生了自然扩容。
输入与D772 S-LoRA 3B W1逐条一致，没有新增权重、负载或重复性能attempt。

| 核查项 | 单轮观测 |
|---|---:|
| planned / arrived / submitted / terminal / response | 各4000 |
| 失败 | 0 |
| 原生生成、adapter绑定和计时重验 | 4000/4000 |
| 原生输出token | 458,224 |
| 完整池准备 / 实际服务逻辑adapter | 500 / 132 |
| 原生worker就绪回执 | 4 |
| E2E / TPOT / 原生重算最大误差 | 全部0 ms |
| 生命周期物理GPU时间 | 8,049.341782 GPU-s |
| GPU-s / binding-valid请求（非独立数值正确请求） | 2.012335 |
| TTFT / TPOT条件 / 联合SLO达成率 | 14.675% / 0% / 0% |

物理时间四个互斥窗口为：业务到达前0、到达窗口7645.369570、drain
372.155451、cleanup 31.816761 GPU-s；合计等于8049.341782。零pre-arrival
只表示业务前60秒尚无GPU持有，不代表CPU准备免费或删除了准备时延。

## 原始时延与可支持的解释

| 指标 | 平均 | P50 | P95（Type-1） |
|---|---:|---:|---:|
| TTFT (s) | 147.224731 | 79.315896 | 557.695343 |
| E2E (s) | 153.300386 | 84.207385 | 563.930895 |
| TPOT (ms) | 53.686196 | 52.988190 | 59.834516 |

平均E2E五段为：提交滞后0.004697 s、提交后至dispatch 39.583550 s、
service TTFT 107.636484 s、decode 6.069960 s、完成通知0.005695 s。
service TTFT包含原生后端排队，不能全部称为prefill计算。

1. 前两个固定500请求phase的平均TTFT为394.492/259.562 s，后六段为
   182.343、116.891、133.856、33.853、29.419、27.383 s。第1000条后
   router queue平均不足0.1 ms，但后端首token等待仍为数十至数百秒。
   因而本轮长尾不只发生在注册前队列，不能仅以网络启动解释所有延迟。
2. 八段平均TPOT约52.821–55.067 ms，超过共同3B阈值29.922/36.216 ms；
   0联合SLO并非仅由启动长尾决定。阈值没有按本系统自身的慢warm结果放宽。
3. 对照同配置W0，W1平均TTFT由38.253增至147.225 s，P95由323.387增至
   557.695 s；这是到达模式改变下的描述性差异，不是算法消融或独立重复。
   总GPU-s较W0低，主要同时伴随更短到达窗口，不能单凭该值称资源效率改善。
4. 本工作点未满足G1/G2共同服务要求。保留未达标行和完整原始指标，不由
   一个静态分片工作点断言Loquetier所有部署不可行，也不为改善排名搜索参数。

下一执行相同配置的W2，仅改变既定adapter轮换映射；不追加重复短回放。

## 远端、资源与正确性边界

- 共同D80只读gzip在通知前发布；本轮实际获取500次，线上1,160,474,851 B，
  本地物化19,482,573,296 B。请求路径打包0，没有模拟等待或本地remote回退。
  获取属于部署准备引用，不伪装为500次逐请求冷miss。实际传输/物化保留计量。
- 服务内存峰值38,909,562,880 B，主机最小可用94,531,792,896 B；服务
  high/max/OOM/OOM kill均0、swap0。磁盘最低142,193,512,448 B，高于运行
  100GiB底线；下一重型启动仍要求150GiB。分析辅助域峰值576,208,896 B，
  high/max/OOM0、swap0，已关闭。
- 服务与watchdog正常终态，GPU context实际释放、服务资源域移除；本轮
  远端两个制品服务及监测已按精确身份停止。没有改动其他项目k3s。
- `workload_passed=true`仅表示规定生成/绑定/计量检查通过。保留
  `n_correct=null`、`independent_lora_numerical_correctness=false`和
  `formal_performance_qualified=false`。D798有限logits对照不扩大为完整
  4000请求、500份独立权重的数值证明；不宣称正式达标或统计显著性。

## 产物与主线

原始结果：`results/ieee_tc/loquetier_qualification/d801_20261024/`。
curated：`paper_results/ieee_tc/baseline_audit/d801_loquetier_3b_w1_result/`。
审计SHA：`9f1962fddd05da651e9c5e37a7b0a4efd6cb86c2b0914cc069c8fe4157adcf06`。
SLO SHA：`d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`。
native配置SHA：`3b5a0274271a95965317bc6bf803728e4eb246d6026ab4338bb640dfc7bff800`。

采用analyze-results区分观测、解释和主张边界；按单轮协议不输出CI，按最新
执行指令暂不绘图。保存完整证据后，仅回收经现时双边SHA与引用核查的重复
下载缓存；唯一工件、负载、失败和旧投稿结果保留。结果检查点用github-sync
的明确路径、diff/来源/敏感内容检查备份，不包含用户dirty manifest。

完整主线仍为同配置Loquetier 3B W2 → 隔离HydraServe 3B → 完整比较与
PrimeLoRA优化 → 同版Full/A1–A5/S1–S13、成本/CE/Pareto、两轮审稿文档、
图表和最终V2同步。HydraServe已完成的平台隔离不重新适配，只处理实际3B
模型数学与加载差异；新静态审计文档不能代替其尚未完成的运行结果。
