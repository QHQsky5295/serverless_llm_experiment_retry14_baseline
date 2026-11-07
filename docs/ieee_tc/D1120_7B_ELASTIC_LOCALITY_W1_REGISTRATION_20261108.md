# D1120：7B 当前版本 ElasticLocality W1 复合消融

2026-11-08，启动前登记，尚未运行。承接 D1119 已封存的 ElasticOnly；
不重跑 Full、基线或3B，不新增负载、工件、性能参数搜索或生产算法。

## 问题、预期与比较边界

本项回答：普通 GPU 命中优先加负载回退，能解释多少资源与服务表现？
比较同一版本的 Full D1115、ElasticOnly D1119，以及本次 ElasticLocality。
相对 Full 是复合消融；不能单独证明 confirmed-state propagation，
LastKnown/A-A/shadow/S2 的既定问题仍保留。

当前 Full/ElasticOnly 联合 SLO 分别98.55%/99.675%，物理GPU时间
7603.874677798245/8575.626704086084秒；Full少占11.3316%，但尾延迟更差。
两点均不在冻结W1 G2预算内，不与cap3的另一工作点拼接。两点远端获取均132次、
131480060线上字节，不能预设本项或Full靠减少下载获益。

若二元locality有用，应观察到真实selector选择变化及相应的GPU时间、
共同SLO和尾延迟后果；命中率提高不自动等于性能改善。负载集中抵消收益、
无明显差别或Full较差都保留。不预设必须改善的百分比，也不改评价阈值、
加入延迟、削弱正常缓存或重复同键直至获胜。首次有效完整观测即封存。

这是既定A1问题，优先级1。只增加一个7B W1执行键，估计约40分钟
wall-clock、约2–3 GPU-hours，仅为排程估计而非结果。复用两个参照，
不加100/1000请求GPU smoke，也不取消其他A1–A5/S1–S13工作。

## 配置与实际选择规则

配置：`configs/ieee_tc/20261108_d1120_7b_elastic_locality_w1.yaml`，
SHA `af32ca772e43fc20546be23f52b744180710257887cf7730c6c2976732939cf1`。
由D1119派生，只将以下目标规则改为既有ElasticLocality，并隔离结果/缓存路径：

```yaml
resource_coordination:
  ieee_system_variant: elastic_locality_v1
  ieee_selection_strategy: gpu_locality
  locality_load_slack: 1
```

复用D448预声明和D450–D453/D1061的slack=1，不重新搜索，不称最优slack。
live可行集最轻载副本为m，GPU-hit子集最轻载副本为h；仅当
q(h)≤q(m)+1时选h，否则选m。q是admitted requests，平局按last-dispatch、
replica ID。HOST/NVMe/Remote统一为非GPU-hit；不按D/T/O、pending load、
GPU utilization或handoff偏好排名。与ElasticOnly的Q元组
(admitted,pending,GPU utilization)不同，因此不把二者说成严格只多一个bit。

与Full共享模型、后端、原生按需缓存、物理可行性/reservation/引用、
1–4实例autoscaler、profile、服务资源包络和输入；关闭handoff、主动
hierarchy、forwarding和软admission，文件层使用demand LRU。
保留实际快照收集成本，不冒称观测已最小化。非目标YAML归一化严格相等，
生产源码没有修改；没有向7B注入3B专用profile或arrival-prefetch。

FP16/TP1、vLLM0.30.0、14slots/seqs、1024batch tokens、context1024、
prefixcachefalse；4000请求、seed42、500逻辑ID、同W1变换、60秒准备通知、
fixed_length_greedy_v1保持不变。cold按真实HTTP获取已发布gzip，保留合法
按需缓存；不按phase清空，不重打包，不把inactive的本地带宽字段误称模拟实验。

## 历史复核、外部依据与最小测试

核对D1061选择规则、D1083消融边界、D1086失败审计和当前router/variant
contract/owned activation源码与git历史。D1086因远端未启动而无效，
不是性能反例；本次D1100既有runner在共同通知前启动本项目7B服务，
校验prepublished_gzip_v1/timing_v2并按InvocationID收尾，不碰共享k3s。

2026-11-08核查
[vLLM 0.30.0按需LRU源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
及[Production Stack负载感知文档](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html)。
前者支持保留普通按需缓存的边界，后者说明缓存偏好与在途负载需要权衡；
后者主要是KV/prefix cache，不是本项目LoRA规则、slack=1或性能优势的证明。
没有移植其评分公式，也不冒充复现该系统。

使用ablation-planner复核既定设计，指定模型不可用，记
`[pending external review]`：仅本地复核，不声称独立审阅完成。
run-experiment沿用现有tmux入口；github-sync进行明确文件备份。
23项CPU测试通过：两套消融非目标配置相等、明确组合身份、禁止主动准备、
实际owner/激活路径、GPUhit/过载回退、可行性、稳定tie-break及快照不变性。
测试采用3/4GiB、swap0、显式taskset CPU2,3,26,27，无GPU性能执行。
shell语法和diff检查通过。完整回放后仍须核验真实机制执行，不用测试代替。

## 执行、保护和完成条件

唯一tmux名：`tc-d1120-elastic-locality-7b-w1`。启动前备份，然后新鲜核验
空闲GPU、至少102GiB可用内存、至少150GiB可用磁盘；32GiB预计增长沿用
现有入口。服务72/80GiB、swap2；辅助3/4GiB、swap0；CPU隔离不变。
测量时不做维护、压缩、哈希扫描、测试、推送、远端配置变更或其他重型作业。

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1120_20261108/7b_elastic_locality_w1 \
  d1120_7b_elastic_locality_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1120_20261108/7b_nvme_elastic_locality_w1 \
  /dev/shm/tc-d1120-7b-elastic-locality-w1 \
  configs/ieee_tc/20261108_d1120_7b_elastic_locality_w1.yaml \
  tc_ieee_elastic_locality
```

命令登记不是已启动证明；实际启动后记录PID、scope和remote invocation。
运行结束核验物理释放，取同clock远端journal，复用D1113 collector、
D299 review和D455 stages的显式scenario接口。复用D1119/D1057机制分析，
按4000条原始快照重算gpu_locality(slack1)；检查handoff、主动file/GPU/HOST/
residency计划应为0，同时核验正常按需行为。自然first-service按实际epoch
报告，不为凑20样本追加同配置运行。

逐请求与D1115/D1119匹配arrival/prompt/adapter/target/native token合同；
保留同后端输出摘要差异，不将其未经验证归因于舍入。source
n_correct=null及独立数值正确性未证明的限制不隐藏。联合SLO/资源/尾延迟、
阶段、获取/字节、失败全部报告；已知开发trace单轮仅作描述性结论，无CI。

144个旧投稿保护文件、用户dirty manifest及两参照raw SHA已核验不变。
来源SHA、测试日志路径及配置记录于
`paper_results/ieee_tc/ablation/d1120_7b_elastic_locality_w1/prelaunch.json`。
不stage用户manifest、历史dirty状态全文或raw大日志。
PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1/单轮V2及D394共同阈值、D594预算保持原身份。

下一阶段是本键真实完成/分析，再按A1继续单因素及累积消融；不重启
baseline复现、不重复Full、不切回3B。全部A1–A5/S1–S13、CE/Pareto、
两轮审稿文档和最终图表/V2交付仍开放，本次不宣告全目标完成。
