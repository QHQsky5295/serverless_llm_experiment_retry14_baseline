# D799：Loquetier 3B完整静态部署绑定

## 问题与有界处理

D797修复原生注意力对Llama-3.2几何/RoPE的兼容，D798对两个既有真实输入
完成六项模型logits/token对照；两逻辑adapter只有一份独立payload。这不等于
500-ID部署或4000请求的数值正确性。当前任务只接通已验证内核与真实3B
完整静态池，不进行baseline性能参数搜索。

复用原`inspect_static_capacity`读取500-ID冻结清单、张量头和配置，不读
未来热点、不复制权重、不新建trace。结果：350个rank8、150个rank16；
四个静态分片各125个，按原有静态字节平衡规则分配。较大分片的FP16
backbone=6,425,499,648 B，FP32原始PEFT=2,991,063,040 B，FP16 prepared
=2,293,760,000 B，指针=224,000 B；batch4/context1024完整KV=469,762,048 B。
合计12,180,308,736 B（约11.34 GiB），**不包括临时workspace、CUDA与碎片**，
静态结果保留`safe_to_launch=false`，不能冒称实测峰值或容量证明。

## 配置选择与实现范围

历史3B gate `loquetier_llama32_3b_limit256_filtered1024_gate_20260520.json`
SHA `f60586f61fb42f7191f165e28e1a7cda25071003d8dbbc51241adb18e155d2a6`
的batch已经为4；本次保持batch4，不把7B D565推导的batch8移植到3B，
不另开批量扫描。旧gate仅256-ID、512输入上限、2输出，故不能代替本次
完整池、共同生成合同的功能覆盖。

现有服务只作两类必要修改：

- 无容量派生策略时，batch绑定冻结静态计划，不再硬编码为7B初始batch2。
  仍保留原PEFT owners与原生SMLM/异步批处理、四分片、顺序启动、静态路由。
  不采用7B owner-release/capacity证据，也不改变baseline调度算法。
- 显式`model_kernel`绑定不可变构建回执、二进制SHA、Python导入根与实际
  backbone config SHA。Llama-3 RoPE不能无绑定地误用旧内核；默认7B路径不变。
  原生导入后仍核对实际kernel文件与SHA，不覆盖已安装环境。

官方核查：[Loquetier固定版本ModelGeneration](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/src/loquetier_src/model_generation.py)
保留其KV page16、原生请求分组、mixed prefill/decode、greedy与终态释放。
此引用支持实现范围，不作为本机性能或完整正确性证据。

37项CPU回归通过（1.299秒），包括旧7B容量路径、模型专用内核身份、静态
batch约束与已有元数据测试。准备脚本最初有一处括号语法错误，在执行前
即退出；没有启动GPU或修改实验数据，已修正并保存失败/空资源域关闭记录。

## 下一个执行与边界

使用既有`adapter_coverage_v1`：原trace前500条输入与固定排序500个既有ID
的功能映射；不是新性能负载，不冒称W0。相同canonical prompt、原生token、
固定目标、真实远端D80已发布gzip、共同60秒准备与完整资源计量继续生效。
该一次回放同时检查四worker完整池的准备、mixed-rank执行与实际资源。
复用D798与已有服务测量测试，不再追加100/1000小回放或独立容量搜索。

启动前仍执行原内存/磁盘/GPU保护；一次一个重型任务。D799尚未启动GPU时
不提前称覆盖已通过。若出现具体兼容错误，仅有界处理该错误；低SLO本身不
触发baseline优化。随后Loquetier 3B W0/W1/W2、HydraServe 3B、完整比较与
PrimeLoRA优化，再同版Full/消融/机制/敏感性/成本/图表及两轮审稿文档。

原始目录：`results/ieee_tc/loquetier_qualification/d799_20261024`（旧主目录
gitignored results）；curated见`paper_results/ieee_tc/baseline_audit/d799_loquetier_static_binding`。
其他项目k3s不改。147保护清单与三项用户dirty保持不变，图表继续延期。
