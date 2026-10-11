# D446：封存的五系统 7B W0 同合同观测

日期：2026-10-12。本项只做离线关联，不启动推理、不改变任何系统配置、
不重新发布或下载工件，不重复已有性能点。完整计划、执行记录、最新基线
规则、冻结指标 V1 与单轮 V2 已完整读取；共同阈值和 Resident 参考不变。

## 输入与核验边界

复用 PrimeLoRA D371/D397、ServerlessLLM D414、vLLM D423、S-LoRA D435，
加入 Loquetier D445。保留两个未达到共同 SLO 的系统，不按结果筛选。
先核验各自封存 audit/manifest，再关联全部4000条请求的源记录、adapter、
arrival、prompt、原生输入token、target、生成合同、内容身份与实际资源账本。
不把不同后端输出逐token一致当成通过条件，也不把绑定证据冒称独立数值证明。

Loquetier 保留四个125-ID静态分片、原生 SMLM、FP16、batch2，未搜索性能配置。
离线程序明确读取实际 `native_registered_monotonic_s`，同时验证其先于实际
schedule和首token；不套用vLLM字段。原生adapter路径、index、request和
forward-count须匹配。完整静态目标、内容、ready与进程证据复用已封存D445审计。
Loquetier原生batch注册、S-LoRA ModelRPC注册、vLLM引擎入口是不同阶段边界，
阶段统计仅作描述，不能把它们视作完全相同的因果切分。

扩展现有比较脚本；默认旧路径不变，新参数必须保留S-LoRA。输出写入新的
`paper_results/ieee_tc/main_comparison/d446_7b_w0/`，拒绝覆盖。
旧D436四行及相对差、匹配合同SHA必须逐字段一致；Resident参考拒绝重冻结。

```bash
/home/qhq/anaconda3/envs/LLM/bin/python -B -m scripts.ieee_tc.check_d299_full_context \
  --matched-w0-output paper_results/ieee_tc/main_comparison/d446_7b_w0 \
  --include-sealed-slora --include-sealed-loquetier
```

从主worktree、受限CPU-only资源域执行：GPU隐藏、CPU2/3/26/27、
MemoryHigh3GiB/Max4GiB、swap0。48项CPU测试通过（0.272秒）；不是性能重复。
本文件在离线关联前登记，后续实际结果另附。当前不画图。

完成后继续HydraServe及剩余7B比较，再依据G1/G2推进必要的PrimeLoRA优化。
W1/W2/G2、3B、ElasticLocality、dLoRA3B、A1–A5、S1–S13、审稿文档及后续
学术图表全部保留；本次W0不能替代这些实验或宣称完整系统目标已达成。

## 完整关联结果

五系统各4000条请求，458224原生输出token、132实际adapter；共同合同SHA
`384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399`。
旧D436四行和相对差逐字段一致。以下均为已知W0负载上的单轮观测，非独立
留出评估，无运行间CI。GPU-s分母是已核验的协议绑定有效请求数；原始
`n_correct=null`与独立完整数值证明未成立的事实保留。

| 系统 | 联合SLO | GPU-s/request | 平均TTFT (ms) | P95 TTFT (ms) | P95 TPOT (ms) |
|---|---:|---:|---:|---:|---:|
| PrimeLoRA | 97.700% | 3.620631 | 1150.798 | 2837.896 | 51.741 |
| ServerlessLLM | 98.125% | 4.022441 | 973.687 | 1052.246 | 32.929 |
| vLLM | 99.475% | 4.028844 | 358.019 | 1200.507 | 27.177 |
| S-LoRA | 94.425%（未达95%） | 3.789269 | 7396.417 | 17699.067 | 45.953 |
| Loquetier | 4.350%（未达95%） | 4.249596 | 262600.734 | 555110.032 | 68.613 |

1. 在本次已达共同SLO的W0观测中，PrimeLoRA物理GPU占用最低：相对
   ServerlessLLM低9.9892%，相对vLLM低10.1323%。但联合SLO分别低0.425、
   1.775个百分点，P95 TTFT分别为其约2.697倍、2.364倍。因此是资源—服务
   质量权衡，不是全指标领先，也不是跨W0/W1/W2的M1完成或G2通过。
2. Loquetier较长等待主要出现在原生batch注册之前：提交至注册平均
   262387.431ms，注册至首token210.333ms。前一段包含router与原生队列，
   不全是计算或网络；D445进一步分解保留在其报告。本表没有隔离静态分片、
   并发容量或准备策略，不能由whole-system差值直接推出某个机制的因果贡献。
3. S-LoRA与Loquetier都完整列入表，但未满足95%联合SLO，不进入“达标条件下
   最低资源”的比较集合。PrimeLoRA相对二者的原始资源减少分别4.4504%、
   14.8006%；不能写成两套均达标系统之间的G1优势，也不为让其达标而调参。
4. PrimeLoRA、ServerlessLLM、vLLM各132次远端传输/131480060线上字节；
   S-LoRA、Loquetier各500次/420740410字节。前者之间没有W0字节减少证据。
   后者使用完整静态池准备，差异不能替代层级缓存或confirmed-state的隔离实验。
5. 全系统平均E2E依次4825.670、4000.734、3182.814、10962.479、269672.942ms。
   PrimeLoRA终点是控制器完成，其余是外部HTTP完成；JSON另外提供共同的
   arrival-to-last-native-token，不能悄悄把两种E2E终点混称完全一致。

当前结果支持继续定位PrimeLoRA入口前等待及扩容/churn下的资源—时延关系，
但不能仅凭本表判定瓶颈原因或改算法。仍先完成剩余7B资格；随后按历史日志、
原始论文/官方实现及可证伪假设开展PrimeLoRA优化，保留九个公式和冻结评价尺子。

## 产物与资源记录

新目录 `paper_results/ieee_tc/main_comparison/d446_7b_w0/`：

- comparison.json SHA `95a1037c494b87d1c52c151bc01850b5a96c9cf578f6af3d117a40d73e4b6edf`。
- metrics.csv SHA `2e31ef256f1984c2a6fd1a13c31ffcab2e5db6495d288e6a40b739f9b462546f`。
- resident_reference.json与旧D424字节一致，SHA
  `737c84804023c9f98677fb5f19c8240b4b05cabdb52f6ecdbbfd4527964c3269`。
- U_ref=16115.377162149176 GPU-s，Q_ref=1200.5068596918136ms，
  B=12086.532871611882 GPU-s；只属于7B W0，W1/W2尚未标定。
- 48项CPU测试通过，0.272秒；测试域峰值295170048B。完整离线关联一次成功，
  峰值1305673728B。两域high/max/OOM/swap均0，均在真实递归进程树为空和
  InvocationID匹配后停止。没有GPU执行或新的性能数据。
- 独立语义审阅者PASS未提供；本报告不把本地核验冒称外部评审。
- 全部71项分析来源SHA另行复核通过，147个旧保护条目和三份用户修改不变；
  保护核验SHA `1ff66513bbe7dc141d9bd14e43a7ce4f111cee38aeb215db2db8eebe141fb657`。
  核验域峰值29736960B，high/max/OOM/swap0；实际递归进程树及身份核对后已停止。
  原生Loquetier/S-LoRA历史补丁SHA未变，HydraServe checkout仍clean。
