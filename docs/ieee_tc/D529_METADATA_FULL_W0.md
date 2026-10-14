# D529：保留运行计数，验证元数据候选的完整 W0

2026-10-14。沿用 D520–D526 的单一候选，不新增优化假设或参数搜索。
D528 原100请求已完成；本轮不重复短回放、标定、设备微测或基线。

## 依据与唯一新增服务改动

D519 的阶段证据和 D520 的源码依赖分析指向 CPU 已知映射经 GPU 计算再
读回的控制依赖。[固定 vLLM 0.30 原生实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/punica_wrapper/punica_gpu.py)
仍可核对该元数据构造路径。D526 已验证实际安装与5060次标定更新；这些
不能替代 D528 的运行计数。D528只保存安装快照，实际更新次数未知。

因此，仅在已有 `ieee_gpu_reference(operation='release')` 返回的回执中
保存 `DenseMetadataPublisher.snapshot()` 的 CPU 标量。原有owner、lease、
释放操作、完成fence、RPC次数、元数据更新和生成顺序均不改变；默认未安装
publisher时不增加字段。无额外GPU查询、读回、同步、日志线程或profiling。
请求结果既有 `gpu_reference_evidence.release_receipt` 会完整保留该字典。

这些计数是 **worker累计值**，不是请求独占值。按owner/worker身份分组，
使用最后可核验快照作为已观测下界，不逐请求相加；不能由累计更新数直接
推断因果性能收益。失败/缺回执保留，不补零、不以其他运行数字代替。

## 最小验证与配置

- 21项受影响CPU测试通过，0失败/错误/跳过；未初始化CUDA。
- 新测试覆盖默认路径、单次释放/计数读取、快照独立性、原释放失败及计数
  失败传播。使用没有CUDA方法的占位对象，新增设备操作会使测试失败。
- 实际D529 YAML经ScenarioRunner实际子配置路径加载D526的200服务及24准备
  类，不替换为测试专用profile。与D528只差输出/cache路径，数值配置相同。
- 本轮服务源码仅新增回执字段；IEEE九式、缓存、路由、admission、生成、
  批处理及HOST/GPU预算保持不变。
- 测试资源域峰值698589184 B，high/max/OOM/kill/swap为0；实际递归进程为空、
  已关闭且路径消失。原147投稿文件与用户manifest校验未变。

原始：`results/ieee_tc/p2_backend_qualification/d529_20261014/`。
curated：`paper_results/ieee_tc/p2_backend/d529_metadata_full_w0/`。
使用run-experiment/monitor-experiment保持原入口、单次执行和实际释放核查；
github-sync用于经校验的显式路径备份，不提交原始大日志或用户manifest。

## 完整评估与未知项

仅运行一次原4000 W0。沿用现有真实远端只读交付、原trace/adapter集合、
共同60秒准备和冻结V1/单轮V2；D528的100请求不是独立留出证据。
评价先检查原生合同、内容绑定、时间线、生命周期及共同联合SLO，再比较
物理GPU-s与尾部；不据胜负重复同键，不画图、不改变阈值。

D528与D475同后端的10/100输出hash差异仍保留为原因未知。原始记录没有
token数组，不能从SHA恢复差异，也不能用D475/D471已有8条差异或局部
非零LoRA微测推定无害。D529须继续保存同输入/输出hash对照；独立数值
正确性 `n_correct` 不补写，完整运行不是自动完成数值等价证明或最终排名。
本次是有明确证据边界的开发评估，不冒充正式留出或全模型oracle。

完成后：真实释放→远端关闭及双侧记录→既有严格analyzer→运行计数/输出
审计→保存并比较。无收益则归档本候选，不能扫描参数或重复短测寻找胜出。
7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13、成本/CE、审稿文档与最终
图表全部保留。共享k3s不操作；HydraServe D510已完成，不重做。
