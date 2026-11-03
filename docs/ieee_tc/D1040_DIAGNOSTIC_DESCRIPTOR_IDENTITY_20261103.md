# D1040：修正诊断的实际HOST读取身份，不冒充新性能优化

2026-11-03。当前主线仍是3B PrimeLoRA。目标与完整PLAN、冻结V1/单轮V2
已经重读；上一目标轮D1039完成诊断与备份为PROGRESS，但本轮发现其实现身份
与父点不一致，必须限制此前解释。本文件接续交接，不修改用户dirty的
EXECUTION_STATUS.md，不接管共享k3s。HydraServe隔离适配和D810基线证据复用。

## 新证据及对最初判断的纠正

按全部请求流式遍历实际native_host_descriptor回执，不仅检查shell默认值：

| 已有运行 | 请求数 | 缺回执请求 | 实际读取身份 | 回执出现数 |
|---|---:|---:|---|---:|
| D1033完整W2 | 4000 | 0 | fresh_native_tensor_descriptor_v1，D747 SHA | 8016 |
| D1039诊断 | 1000 | 0 | python_tensor_descriptor_v1 | 2006 |

D1033四个请求worker PID为4125618/4129029/4128955/4173524；各有
4039/3917/14/46个回执。D1039对应583631/585940/586060/618154，各有
1021/933/10/42个回执。它们是回执出现次数，不是独立清点调用数；仅覆盖
请求相关worker，不把无请求的额外诊断进程填入此集合。

因此，本轮最初“主系统漏掉历史已接受融合实现”的假设已被完整父日志否定。
run_d1000本身未设置该环境，并不证明D1033启动时也未传入它；D1033文档及
原有native_descriptor.json已经记录实际启用。本次错误是D1039启动命令遗漏
显式环境，不能用恢复开关重命名为新优化。D1039的Python清点99.808%线程CPU、
177.686ms等重叠结果全部保留，但不归因给当前D1033融合路径。

另一个此前已更正的事实继续有效：当前使用LocalSourceReferences进程内路径，
并未启用FileStateProcess。D1039没有文件进程计时不等于文件操作无开销。
本轮不为测量而打开另一种文件执行机制，不再启动未使用的文件观察器。

## 历史与一手源码

D747/D748的融合读取逐次观察当前Tensor/storage/pinning，只缓存函数入口；
D749完整W0曾从3727增至3787联合通过，明确接受为开发父版。该收益不直接
外推到当前W2。D864–866是不同的整图native候选，未被接受，不恢复graph。
D329跨调用缓存、D352同事务复用、D869联合图投影及D1036容量负结果保持原结论。

本轮核查官方[vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
和[PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)。
原生加载/替换和实际storage pinning仍要求读取当前状态，不能由逻辑ID推断
状态未变。这些源码约束观察语义，并不证明本地性能收益。

## 唯一修正与一次有界补测

只扩展现有runner的显式fresh_d747_v1选择和既有observer的D1040身份；
不修改生产调度、清点算法、九公式、原生库或任何性能参数。缺库、SHA/ABI
不符仍由已有loader报错，不静默回退。原来未指定的启动默认保持不变。

复用D747的195688B库，SHA
b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044；不重编译，
不重做9种CUDA pin状态和组件微测。修正后的问题仅是：**当前32条缓存、
真实融合读取下，完整HOST清点的墙钟/线程CPU及其decode重叠占比是多少？**
现有父运行没有fresh_inventory_v1分段，D1039则测了另一条实现，不能代替。

只跑一次源W2前1000请求的诊断修复身份；不是为寻找更好性能重复完整结果。
前500和后500全部保留，不挑快请求、不从延迟中减去观察开销、不推算G1/G2。
D1037仲裁修正已在当前源码，其完整性能尚未测量；不会给它挂上D1033成绩。

配置仅相对D1033改三个独占目录。执行命令：

```bash
FAASLORA_D1000_RUN_ID=d1040_20261103 \
FAASLORA_D1000_RUN_TAG=d1040_3b_native_descriptor_prefix1000 \
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261103_d1040_3b_native_descriptor_prefix1000.yaml \
FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
FAASLORA_D1000_PREFIX_COUNT=1000 \
FAASLORA_D1000_NATIVE_PROFILE=host_spans_v1 \
bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

启动后检查实际worker回执身份，不再只凭命令或YAML一致性宣称一致。
结束→真实释放→复用D1039/D1016/D723分析与已有核心计时器→保存来源和限制。
若清点并非主要开销，就转向既有阶段证据，不继续此局部优化；有明确瓶颈
才登记下一项可证伪候选，不自动追加prefix或容量扫描。

## 资源、封存与完整范围

先无损压缩D1029已关闭的请求JSON：288595652→49752773B，两次解压SHA
均8ca69c1d86b6843e87340f04cce3b05c2b68bf1891f0a071826b63c0492c6e7e；
释放238841856B，回执paper_results/ieee_tc/storage/d1040_d1029_request_lossless_20261103.json。
没有删除唯一权重、负载、原始内容或其他项目数据；原稿保护不变。

启动内存102GiB、磁盘150GiB并覆盖预测增长，service72/80GiB、swap2GiB、
CPU4–23/28–47；辅助3/4GiB、swap0、CPU2/3/26/27；运行底线16/100GiB。
不在测量中压缩/测试/构建/推送/重启远端。沿用3B真实远端已发布交付对象，
不请求时打包、不修改网络或工件。采用run-experiment和github-sync复用
启动/备份流程，analyze-results区分实际观察与可支持的主张；不采用通用
vLLM技能示例阈值替换冻结合同。

主点D1033仍3772/4000=94.30%、8155.283736GPU-s，不满足3800；独立数值
正确性未知，n_correct=null。冻结V1/单轮V2、D648和W2预算均不变。性能
接近目标前不画图，不重启已完成的baseline基础适配。

两模型同最终版本W0/W1/W2/G1/G2、独立数值正确性、最终外部对比、A1–A5
全部消融/confirmed-state/first-service、S1–S13、CE/Pareto/价格敏感性、
B1/B2/C1/C2/C3/新颖性及两轮审稿文档、IEEE图表、最终V2集成全部仍未闭合。
当前仅登记修复后的诊断，尚无D1040 GPU结果；备份分支不冒充已集成V2。

## 测前核验已完成

15项定向测试全部通过，包含实际shell子进程环境、默认不变/未知值拒绝、
配置仅三目录变化及既有loader/图语义检查；没有重复CUDA资格或组件微测。
现有实际库在当前Torch2.13/CPython3.12加载并返回正确SHA和descriptor模式。
最终辅助域峰值316260352B，high/max/OOM/swap均0；147项旧投稿保护及两个
用户dirty文件SHA未变。资格与两轮全部请求回执审计保存在
paper_results/ieee_tc/p2_backend/d1040_diagnostic_descriptor_identity/prelaunch/。

第一次离线封存因glob把压缩回执sidecar也当请求结果而停止；当时15项测试
已通过、GPU未启动。保留原prepare.py与测试日志，resume只修正结果文件
选择并复用已完成测试，不覆盖旧证据、不当作一次性能失败/重跑。

18:08远端认证health正常，clock ID仍为b7a04b62f38c4ae6adabb4da5c0fd8dc，
prepublished_gzip_v1/artifact_timing_v2。未重启服务或修改交付配置。

提交前仅删除新YAML末尾一个空行，测试时SHA
a5ebe34c05d880a97708205bffe7672748be72cd47574b6b8feda0befb53996c保留在原资格
记录；新文件加回一个换行字节与该SHA一致，证明无配置语义变更，不重跑测试。
