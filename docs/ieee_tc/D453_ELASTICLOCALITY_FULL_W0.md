# D453：ElasticLocality 单轮完整7B W0

2026-10-12，首次运行前登记。依次通过D450原100、D451全500-ID功能覆盖、
D452原1000兼容检查后，执行计划§6.7的原4000请求完整回放；不是重复资格点、
不是对照性能调优，也不重跑已封存的五系统W0。

## 问题与冻结合同

研究问题：与PrimeLoRA共享后端和autoscaler的“普通GPU局部性＋负载回退＋
按需文件LRU”在相同完整负载下的资源—服务质量表现是什么？这是强内部对照，
不是最近邻论文的冒名复现，也不是能单独隔离confirmed-state作用的单因素实验。

- 复用原4000行W0，无新trace/工件/权重/抽样。500-ID可访问全集中实际出现132个
  逻辑adapter，458,224目标输出tokens。与D435及D446保留输入合同核对。
- 原源SHA efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c；
  PREFIX_COUNT清空，full_source_v1、原始open-loop到达、不平移起点。
- D453配置与D450/D452除输出/NVMe/HOST路径外完全相等。slack1预定义、无搜索；
  同vLLM0.30.0/FP16/TP1、slot/concurrency14、最大4GPU、原autoscaler与物理保护。
- 无主动handoff/hierarchy/coordinated admission；不把原生按需加载、安全准入或
  普通缓存删掉。IEEE Full代码及九个公式不改动。
- fixed_length_greedy_v1、相同prompt/tokenizer、原生token计数；共同60秒部署通知。
  资格保护1800秒不变，保留开发/资格身份，不改称独立留出正式测试。
- D78/D80只读gzip交付缓存复用，真实HTTP，无请求内打包、人工限速或模拟等待。
  读取/传输/竞争/物化保留计时，不从结果中扣减累计span。

## 判定和必须保留的结果

完整正确执行需4000/4000原生数量、adapter/source/lease与输入绑定、零隐藏fallback，
可靠时间分解、实际GPU allocation/release和两端HTTP证据。未知的原始n_correct
字段保留未知；协议绑定验收不能冒称独立kernel数值证明。

读取冻结V1＋single-run V2，采用D394共同SLO，至少3800/4000联合达标后方能进入
W0达标资源比较；不达标仍报告所有原始指标和物理GPU-s。CE不作为排序主指标。
一次有效运行即封存，无CI或重复挑优。完整W0不代表跨W1/W2的G1/G2已完成。
activation触发、engine-ready、取消和first-dispatch分别计数；未触发机制不归因。

## 启动、资源与收尾

20项既有独立回放测试通过（0.678秒），非目标配置相等、输入逐条相等、代码与
D452备份594c3e0相比未改动。仅现有launcher新增d453固定分支和新的路径配置。
147项历史保护及用户文件、PLAN/V1/V2保持不变。新原始目录拒绝覆盖已有运行。

重新检查磁盘≥max(150GiB,100GiB+1.5×32GiB)、主机内存剩余增长量＋18GiB、
空闲GPU、已推送HEAD和来源SHA。服务72/80GiB、swap2GiB，辅助3/4GiB、swap0，
CPU分离、实际workers受控、独立watchdog；一次仅一个重型任务。
启动前记录实际链路和远端身份。运行中不做构建、测试、哈希扫描、清理和远端维护。
完成后确认实际GPU释放，才按InvocationID关闭归属明确的远端服务及空辅助资源域；
下载记录核验SHA，独立输入/机制/计量校验，再保存完整数据和备份。暂不画图。

配置：configs/ieee_tc/20261012_d453_7b_elastic_locality_w0.yaml。
输入证据：paper_results/ieee_tc/elastic_locality/d453_full_input_qualification.json。
raw：原项目results/ieee_tc/elastic_locality_qualification/d453_20261012。
运行：现有scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh d453。

## 后续主线（全部保留）

本轮校验封存后更新同合同7B W0比较，再完成7B W1/W2参考、比较和G2、必要的
PrimeLoRA优化，随后3B。HydraServe原生资格仍受现有Kubernetes访问前置条件限制；
不改未知集群、不以裸vLLM替代。dLoRA 3B、A1–A5、S1–S13、CE补充、两轮审稿
设计文档及后期图表仍按完整计划执行。不能因局部可行性宣称整个目标已完成。

## 单轮实测与证据边界（2026-10-12 收尾）

同一次运行完成4,000/4,000请求、458,224个原生输出tokens。原生数量、
source/adapter/lease绑定和既有D435逐请求输入合同全部匹配。未知的原始
n_correct不回填；这不是独立kernel数值正确性证明。

| 指标 | 单轮观测 |
|---|---:|
| 冻结共同联合SLO | 3,831/4,000（95.775%） |
| 生命周期物理GPU-s | 9,892.715870 |
| GPU-s/协议绑定请求 | 2.473179 |
| 平均TTFT | 1,003.697865 ms |
| Type-1 P95 TTFT | 2,420.607829 ms |
| 平均dispatch等待 | 528.828451 ms |
| 平均service TTFT | 474.869414 ms |
| 平均/P95 TPOT | 29.212305 / 54.523438 ms |

这次资源和尾延迟观测低于已封存PrimeLoRA W0，而联合SLO低于其97.70%。
保留该不利于Full资源优势的证据，不因结果重复运行或调整对照。
需要在同合同六系统分析中核验同后端输出差异；目前封存包仍明确标记
ranking_eligible=false，不能直接据此宣称完整跨场景优劣或因果机制。

无主动准备计划，普通文件LRU驱逐未触发。4,014次路由选择尝试含2,710次
GPU-hit、185次slack回退、1,119次无可行GPU-hit；不是4,014条完成请求。
实际dispatch tier：remote132、GPU2706、HOST315、NVMe847。
natural_scaleout共33次触发，其中30次ready、3次取消；另有1次initial。
first-service须另按epoch第一条dispatch绑定，不能用触发数替代。

132次真实HTTP/131,480,060线上字节，两端ID/SHA/字节核验一致；请求内
打包为0，D78/D80既有缓存直接复用。4040次资源采样，峰值20,717,527,040B，
high/max/OOM/swap均0。实际GPU释放先于远端服务关闭，清理及下载SHA通过。
原始运行、失败/取消事件及全部历史结果保留；没有增加重复、权重或负载。

完整证据见paper_results/ieee_tc/elastic_locality/d453_actual_full_w0.json。
