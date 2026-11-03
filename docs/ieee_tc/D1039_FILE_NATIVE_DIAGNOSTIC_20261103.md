# D1039：当前3B文件准备与原生HOST清点的有界诊断

2026-11-03，接续D1038；完整目标仍active。这是测量接入及一次预登记诊断，
不是已经获得性能改进。EXECUTION_STATUS.md为用户dirty，继续不编辑、不暂存。

## 主线、问题和停止条件

已重读目标、完整PLAN、执行状态、冻结V1/单轮V2和D1038。上一目标轮为
PROGRESS（保存了已有证据与未决问题），本轮不重复纯文档推演。当前父配置
是D1033的32条CPU/GPU缓存、concurrency8、token4096、1–3实例，原生合同
4000完成、联合3772/4000、8155.283736 GPU-s，尚不满足3800，数值正确性
仍未闭合。D1036的110条缓存负结果保留，拒绝重新进行容量网格。

D1038的远端关联中，违约请求的文件发布包络均值461ms、发布后到native load
约412ms；它们不是纯CPU、锁或网络时间。待判定的两个问题：

1. 当前缓存填充后，文件命令的提交→执行、实际执行、执行→首次消费回执
   各占多少？重点分类publish_transfer、preparation_snapshot、acquire_best等，
   不将重叠命令包络相加作为请求延迟。
2. 当前native控制事务中，完整HOST清点占多少线程CPU/墙钟？使用原有
   fresh_inventory_v1，不以旧D863/D868配置替代当前实测。

固定复用原W2完整映射验证后的前1000条，不生成新trace/权重；分析时另列
ordinal>=500部分，但不删除前500条，不把诊断变成完整G1/G2。一个唯一执行键；
截断/缺失如实报告，不为凑样本追加相同运行。结束先确认资源释放，再按测量
决定单一优化方向；不能把诊断计时从旧主结果中减去来伪造新结果。

## 接入与测量语义

- 复用FileStateProcess已有_execute_timed和FileStateCall首次观察边界。
  默认关闭，仅command_spans_v1及既有nonformal/service/prefix门禁可启用。
- 每个父进程的文件日志上限64MiB/200000条，独占创建。只保存operation、
  owner/worker/sequence、成功标志和时钟；不保存命令参数、返回对象或权重。
- 提交→执行包含序列化、IPC、队列和调度；执行是完整owner命令；执行结束
  →首次观察包含IPC、调度及迟加入。不能分别冒称纯队列/纯IO/纯IPC。
  worker_cpu是整个进程CPU（含并行SHA线程），不与墙钟相加、不要求小于墙钟。
- 首次回执记录一次，包括失败、取消后重入；正常关闭记录提交/观察/保存数量。
  日志写入失败不得阻断已完成操作的引用/结果接管；关闭时报告诊断失败，
  不生成完整footer冒充成功。突然退出、未join和截断保持不完整身份。
- native host_spans_v1 + fresh_inventory_v1复用已有实现与分析器；不启用
  Python栈或step子组件。HOST清点嵌套于控制事务，不能再次叠加；主机span
  不是CUDA kernel时间。观察开销保留，不宣称与无观察Full的因果性能差。
- 不改IEEE公式、在线策略、远端配置或vLLM源码。源码包含已提交的D1037
  overload优先级修正，故不能把D1033旧成绩挂到本版本。

本轮使用run-experiment复用启动与资源保护，analyze-results限制解释范围，
serving-llms-vllm核对原生诊断边界，github-sync保存可回退检查点。技能中
通用吞吐/TTFT示例不替代本项目冻结阈值，也不引入调参或依赖升级。

已联网核查[Python 3.12 time文档](https://docs.python.org/3.12/library/time.html)：
本机跨进程单调时钟可用于上述差值；process CPU和thread CPU定义不同。
[vLLM 0.30.0 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
在输入处理后执行engine step；本地已有atomic_control_yield，不能重做已关闭
的“排空全部utility导致饥饿”修复。官方结构支持测量边界，不证明本机因果。
历史D329跨调用图缓存、D628/631移锁hash、D864–866、D998等排除项见D1038，
没有本轮新证据前不重新实施。

## 唯一执行身份

配置：configs/ieee_tc/20261103_d1039_3b_file_native_prefix1000.yaml，SHA
6dc73cf4aebfd903fb409f5cc52d42c4b4eb1617f4c120f5b4fc2fdaba0355ac。
相对D1033仅三项输出/缓存路径变化；保留4000源合同，由现有prefix入口投影。

```bash
FAASLORA_D1000_RUN_ID=d1039_20261103 \
FAASLORA_D1000_RUN_TAG=d1039_3b_file_native_prefix1000 \
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261103_d1039_3b_file_native_prefix1000.yaml \
FAASLORA_D1000_PREFIX_COUNT=1000 \
FAASLORA_D1000_NATIVE_PROFILE=host_spans_v1 \
bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

原始目录：/home/qhq/serverless_llm_experiment/results/ieee_tc/
p2_backend_qualification/d1039_20261103/d1039_3b_file_native_prefix1000。
tmux计划名称tc-d1039-file-native；不重用同键、不使用force。
当前为启动前登记，尚未写入GPU运行结果。

## 测前验证、资源和保护

24项不同的定向CPU测试通过（20项首轮；最终9项包含5项复查），覆盖默认关闭、
真实spawn、成功/原异常、取消后join、只写一次、日志边界、写失败后的结果接管、
HOST观察保真及配置仅路径变化。最终增加submitted_commands后的journal用例
已重跑。bash -n、git diff --check通过。测试在辅助3/4GiB、swap0、CPU2/3/26/27，
CUDA隐藏；不是GPU性能/数值正确性验收。未增加广泛资格矩阵。

独立旧投稿seal147项通过，用户manifest及EXECUTION_STATUS的SHA仍分别为
3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df、
41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374。

磁盘原可用161052160000B，略低150GiB。用已有压缩器无损归档D1033已关闭
request result，287843682→49664583B，两次解压SHA均为
1eeeb5f8613ab5baaf5b02df9df84de10ffb4aed61da09795656db91c346c6fd；
释放238178304B，可用161290317824B。回执：paper_results/ieee_tc/storage/
d1039_d1033_request_outcome_lossless_20261103.json。其余项目/唯一数据未删除。

推理service72/80GiB、swap2、CPU4–23/28–47，aux3/4GiB、swap0、CPU2/3/26/27。
既定启动102GiB内存、磁盘max(150GiB,100GiB+1.5×32GiB)，运行底线16/100GiB
及watchdog保持。测前GPU compute为空；启动再次由现有launcher核验。不在GPU
回放期间压缩、构建、备份、远端重启或进行额外传输压力测试。

17:22远端认证health通过，复用PID1984520、Oct31启动的既有3B服务，
prepublished_gzip_v1/artifact_timing_v2及clock ID与D1038一致。两端实际路由
NIC eno1np0/eno1均1000Mbps/full；不声称应用吞吐达到此值。最初enp*查询
无匹配，随后按ip route确定实际NIC；没有网络调整或服务重启。
共享k3s不属于项目，不修改；HydraServe隔离适配与D810基线证据继续复用。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
D648共同阈值仍不变。

## 完整剩余范围

两模型最终版本W0/W1/W2与G1/G2、独立数值正确性、最终外部对比、A1–A5
全部消融/confirmed-state/first-service、S1–S13、CE/Pareto/价格敏感性、
B1/B2/C1/C2/C3/新颖性与两轮审稿文档、达标后IEEE图表和最终V2集成全部保留。
此次诊断不关闭其中任何性能任务，不重新启动已完成的基线必要适配。
