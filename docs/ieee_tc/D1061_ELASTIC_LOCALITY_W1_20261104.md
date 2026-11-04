# D1061：3B W1 ElasticLocality 复合消融

2026-11-04。本项承接 D1057 完整结果，不是新的 Full 性能优化，也不是外部
baseline。上一 goal turn 为 progress：ElasticOnly 4,000 请求及同输入比较完成，
提交 `45a39118be4d01c8e64d5ef28b90dda760801967` 已核实备份到 origin V2 和
FaaSLoRA 独立备份分支；不要重复分析/启动 D1057 或 D1047 Full。

## 科学问题及当前证据

问题：普通 GPU 命中优先、负载回退和按需缓存，是否已能解释 Full 的服务/资源
表现？相对 ElasticOnly，本项增加二元 locality 排序；相对 Full，它是同时去除
主动 handoff、分层驻留/软 admission 和细粒度服务成本排序的复合消融。
不能独自隔离 confirmed-state propagation，也不宣称来自某篇外部论文。

D1057 中 Full 为 95.90% 联合 SLO、6182.715971 GPU-s、1114.091001 ms P95
TTFT，ElasticOnly 为 98.20%、6032.623108 GPU-s、695.139877 ms。两点全部保留。
它们各有132次远端 fetch；Full 较高命中率没有带来净服务收益。下一项不通过
改变 Full、削弱按需缓存、挑选请求或增加同键重复来改变这个事实。

若二元 GPU locality 有益，应实际观察到路由选择变化及其服务/资源后果；
若负载集中抵消了命中收益，也如实保留。仅有命中数变化不是机制收益证明。
本项继续补齐既定 A1，不重新设计消融，不开展无限参数搜索。

## 实现复用与固定选择依据

现有 `Router.select_locality_snapshot`、`elastic_locality_v1`、demand LRU、
owned IEEE activation/引用/物理资源路径均已存在；本轮不改生产代码。
新配置从当前 D1057 3B YAML 派生，除三处独占路径和准确 scenario 名，仅改：

```yaml
resource_coordination:
  ieee_system_variant: elastic_locality_v1
  ieee_selection_strategy: gpu_locality
  locality_load_slack: 1
```

沿用 [D448 的预声明](D448_ELASTICLOCALITY_ROUTING_BOUNDARY.md) 和 D450–D453
实际记录：slack=1 个已接纳请求，不执行旧版性能搜索，不声称此点为最优 slack。
该选择早于当前 3B 消融结果；不从 D1057 的胜负反向选择。原生实现先在 live
可行集合中选最轻载副本 m，在 GPU-hit 集合中选最轻载副本 h；仅当
q(h)≤q(m)+1 时用 h，否则用 m。平局保持 last-dispatch/稳定 ID 次序。
HOST/NVMe/Remote 都只是非 GPU-hit；不使用 D/T/O、pending-load、GPU utilization
或 handoff 偏好排序。保留共同快照/来源收集成本，不能冒称观测已完全最小化。

再次核对历史实现与当前官方资料：
[vLLM Router 源码](https://github.com/vllm-project/router/blob/main/src/policies/cache_aware.rs)
在缓存选择与负载不平衡回退间区分路径；
[Production Stack 文档](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html)
说明缓存收益应与在途负载权衡。二者主要涉及 prefix/KV cache，不是本项目
LoRA 规则或 slack=1 的最优性依据。本轮未移植其公式，没有将 ElasticLocality
冒名标成这些系统；在线主分支来源仅是 2026-11-04 的设计参考。

## 保持不变及最小验证

- Full 对照仍为 D1047。3B 模型、vLLM0.30.0、D1031 profile、FP16/TP1、
  32 LoRA slots、c8、4096 batched tokens、1024 context、0.72显存比例不变。
- 1–3实例 autoscaler、真实 allocation/release、请求驱动来源准备、原生缓存、
  物理 admission/reservation、取消及引用保护均与 D1057 保持一致。
- 远端复用 D80 只读已发布 gzip；请求触发读取/传输/物化保留在时延中。
  本轮不打包、不重启远端、不调整带宽，不生成权重或新 trace。
- 原4,000请求 source42 和同 W1 arrival/adapter/prompt/target；部署通知提前
  60秒，8×500、phase间30秒，不清各phase缓存、不强求固定自然扩容数。
- V1/单轮V2、D648 SLO、D657预算不改。旧终端5000ms不是最终共同阈值。
  n_correct=null、独立非零 LoRA 数值证明未知、开发负载及单轮限制保留。

已按源码 SHA 复用 D1057 的20项入口/实际控制/引用检查及25项统计检查。
另外仅运行既有 BinaryLocalityRoutingContract 的12项 CPU测试，全部通过：
可行集、命中/过载回退、slack边界、稳定tie-break、状态不重复更新、身份与计数。
新配置与 D1057 去除上述目标差异后完全一致，完整组合身份检查通过。
CPU测试不等于真实GPU机制触发或数值性能证明，不增加无理由500-ID或smoke。
测试资源域已退出，3/4GiB、swap0、CPU2/3/26/27；保存时峰值613404672B，
high/max/OOM/kill/swap事件零。来源和复用依据保存到 prelaunch。

## 唯一执行键与完成标准

配置：`configs/ieee_tc/20261104_d1061_3b_elastic_locality_w1.yaml`。
raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1061_20261104/`。
curated：`paper_results/ieee_tc/ablation/d1061_3b_elastic_locality_w1/`。

在配置/证据备份及新鲜资源/远端健康核查后，复用现有入口启动一次：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
tmux new-session -d -s tc-d1061-elastic-locality-3b-w1 \
  env FAASLORA_D1000_RUN_ID=d1061_20261104 \
      FAASLORA_D1000_RUN_TAG=d1061_3b_elastic_locality_w1 \
      FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1061_3b_elastic_locality_w1.yaml \
      FAASLORA_D1000_SCENARIO=tc_ieee_elastic_locality \
      FAASLORA_D1000_WORKLOAD=W1 \
      FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

本段先登记命令，不是运行成功或完成回执。按 run-experiment 使用 tmux；
沿用已批准的新后端与现有受限入口，不执行技能中历史旧环境示例。
服务72/80GiB、swap2、CPU4–23/28–47，辅助3/4GiB、swap0、CPU2/3/26/27。
推理机150GiB磁盘/102GiB空主机内存启动规则及100/16GiB停止规则不变。
未达95%不自动终止；不与压缩/构建/重型分析/推送或远端维护重叠。

运行后：实际资源释放→复用 D1057 collector，显式传入
`scenario='tc_ieee_elastic_locality'`→核对4000输入/原生token/时钟/物理GPU时间→
按每条原始快照重算 gpu_locality(slack=1)，核验主动计划为零及真实按需行为→
与同一个D1047 Full、已完成D1057比较→保存全部结果与负例→备份。
不把路由重试次数当请求数，不把同状态 shadow 选择差异当反事实时延。

当前没有 D1061 性能结果。独立数据目录拒绝覆盖；147项历史投稿保护清单及
两项用户dirty文件继续原字节保留。EXECUTION_STATUS头部是旧D998且含用户修改，
不覆盖不暂存；本页承担当前交接。actual FaaSLoRA V2分歧不强推，备份不等于
最终集成。完整目标 active；LastKnown/NoHOST、7B A1/累积节点、受控A2–A5、
S1–S13、M2资源配置、CE/Pareto、两轮审稿文档/公式解释、IEEE图表和最终V2
全部未删减。按最新要求，当前先留数据表，不生成目标尚未达成的优势图。
