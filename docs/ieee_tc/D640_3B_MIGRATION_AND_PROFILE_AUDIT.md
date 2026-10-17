# D640：3B主线恢复、资产复用与当前实现资格

2026-10-17。完整PLAN、最新基线停止规则、冻结V1及单轮V2继续生效。
本次是有界离线核查和下一项资格准备，不是新增性能结果。7B按D639封存，
不重新打开W0/HF全等分支。使用analyze-results区分观察、解释与下一验证；
experiment-audit依赖的共享规则文件在本机缺失，故不声称完整独立审计通过。

## 已核实事实

| 项目 | 原始证据或本次核查 | 可得结论 |
|---|---|---|
| D138 3B Full | 4000终态、4000原生数量合同匹配 | 历史完整执行，不等于当前配置合格 |
| 平均/P95用户TTFT | 12588.6266 / 45713.6880 ms | 明显存在长等待，不能关闭3B优化 |
| 平均dispatch/admission等待 | 8540.7982 ms | 需要检查控制/准备，而非仅更换GPU内核 |
| 平均service/native TTFT | 4047.8283 / 581.0569 ms | 存在推理外开销；这不是完整因果分解 |
| 物理GPU时间 | 15861.3305 GPU-s，原n_correct=null | 保留原资格未知，不补造正确分母 |
| 静态adapter配置 | 本次逐文件SHA验证500/500；rank8=350、rank16=150 | 最大rank16有静态依据 |
| 权重身份 | 冻结清单内两种adapter_model.safetensors SHA | 500逻辑ID不等于500独立权重；本次未重新哈希全部权重字节 |
| D89初始化 | 24文件内容类、40服务类、96准备类 | 可复用选择索引；旧测量不能冒充新实现时延 |
| 共同3B阈值/预算 | 本次curated清单未找到当前共同warm/Resident合格参考 | 不能借用7B数值或把开发5000ms当正式SLO |

D138配置为8 lanes、最大rank64、gpu_memory_utilization=.72、batch-token预算4096。
其执行commit为2608d027d82df5a79719153e114ff06a7a70679b。D89来源更早。
当前代码已经包含7B阶段验证过的原生状态、文件所有权和控制路径改动；
本次保存实际相关git历史，不由“后端版本同为0.30.0”推断执行合同相同。
旧Full与旧profile判为R2；保留数据与诊断价值，不混入当前主表。

上述等待跨度可能重叠，不能把runtime-slot、ingress、RPC与dispatch均值相加。
旧结果未正式满足共同指标，不从平均时延反推联合SLO、数值正确性或资源最优。

## 当前资格候选及依据

新增独立配置`configs/ieee_tc/20261017_d641_3b_native_source.yaml`，仅用于迁移资格。
保留3B原8 lanes、.72显存比例、4096 token预算、FP16/TP1及已有HOST预算。
不复制7B的14 lanes、min2或其他按7B选择的工作点。

- 最大rank改为静态池实际最大值16。vLLM官方建议按实际最大rank配置；0.30.0
  源码的LoRA stacked张量维度包含该上限。可推断减少不必要容量，但**未测前
  不声称TTFT、总显存或GPU-s按相同比例改善**。
- 显式开启V1 chunked prefill，保留4096预算，使用已有FLASH_ATTN/采样器与
  当前已验证的控制、文件请求所有权策略。官方说明V1通常默认启用chunked
  prefill，并指出token预算影响TTFT/ITL取舍；不凭经验直接套用7B的1024。
- 这是当前实现的迁移配置，包含多个已说明差异，**不是rank或chunking的
  单因素消融**。九个IEEE公式、资源安全上限和评价尺子不改变。

原始来源（本轮联网复核）：

- [vLLM LoRA配置建议](https://docs.vllm.ai/en/latest/features/lora/#configuring-max_lora_rank)
- [vLLM 0.30.0 LoRA配置源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py)
- [vLLM 0.30.0 stacked层源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/layers/column_parallel_linear.py)
- [vLLM 0.30.0优化说明](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/#chunked-prefill)

## 最小补检和复用边界

已有D88的24个代表、46 waves、368个请求case及输入分桶逐项保持。
只保存小型索引变换，不新造请求、prompt、权重或完整trace。source trace SHA
为4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d。

1. 已完成：现有`source_profile_inputs`在禁止CUDA、无HTTP请求条件下解析通过；
   24身份、46 waves、368 cases；bash语法通过。没有把CPU检查称为GPU资格。
2. 下一D641：复用`run_d298_capacity_source.sh`和`backend-model-check`执行一次
   当前3B native source matrix，真实远端已发布交付，单GPU原生计量和清理。
   包含remote/NVMe/file-HOST/native-HOST/GPU路径；后两种HOST表示不混称五层论文设计。
3. 移除新索引中旧Full/admission绑定，因为旧model identity与新候选不同。
   D641是source-only资格，**不能直接标为Full/admission时延profile**。
   从其实际完成长度建立新初始化，再补精确Full模型身份下的必要profile；
   不放宽既有identity检查或使用猜测长度补缺。
4. 随后共同3B warm阈值、当前Full短前缀和完整W0/W1/W2，按G1/G2推进。
   使用已知seed42身份，不称独立留出；一个冻结性能键一轮，保留失败/退化。
   旧3B基线先逐字段复用，受影响才重跑，dLoRA仍仅3B W1/W2。

验收关注原生token数量、adapter/content/slot绑定、真实源tier、KV/slot容量、
完成/取消/释放及实际准备成本。只有具体现象才新增有界诊断，不展开全量HF
逐token相等作为新的强制门槛。368个profile case不是主4000请求性能点，也
不是368个独立运行；原3轮wave设计仅作初始化测量，不重新引入性能CI。

## 存储、安全及交付

仅将已明确释放的D634失败和D636成功main_outcome.json无损压缩，保留各自
pass身份，源SHA与两次解压SHA一致后移除未压缩重复件。回收540532736字节；
回执`paper_results/ieee_tc/storage/20261017_d640_closed_outcomes.json`。
源文件现在从原路径旁gzip恢复/流读，不能因文件名变化宣称历史丢失或重跑。
147投稿保护文件及用户manifest未改。没有清理唯一权重、负载、环境或其他服务。

本轮分析与压缩资源域均限定3/4GiB、swap0、辅助CPU集；实际关闭证据归档。
重型任务前仍须重新验证102GiB可用内存及150GiB磁盘门槛；本次检查不是未来保证。
没有改动非本项目k3s。HydraServe已完成的隔离适配继续复用，不再争用该集群。

3B全部系统、A1–A5、S1–S13、离线成本/CE、两轮审稿说明及最终图表仍全部ACTIVE。
按最新指令，本阶段不绘图、不改论文文件。此核查不意味着3B达标或整个计划完成。
