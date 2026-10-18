# D670：已验证HOST图的独立复制组件核查

2026-10-18，3B优化主线的有界CPU组件实验；尚未修改生产代码、运行新GPU
回放或接受性能候选。D665完整W0联合88.75%仍未关闭，D668是定位诊断。

## 假设、历史与来源

D668实际测得HOST清点平均69.830ms、CPU/wall99.21%，但source_snapshot
父操作还含其他工作。当前无staging分支对刚清点完成的图调用通用deepcopy；
不能由父子差值把其余时间全部归于复制。D202已经省去tensor-view描述并保留
真实图，其旧组件记录仍显示复制有成本。本轮不重复D202投影或降低物理检查。

直接让native_footprints与native_staging_footprints共享嵌套可变记录，会违反
既有返回值隔离测试，所以不采用。候选仅按现有已知结构复制所有dict/list，
复用不可变标量，保持完整字段及独立可变对象。输入仅为本次新鲜生成、无view/
无staging的registered图；未知结构明确拒绝，不回退为未经核验的浅拷贝。
每次live membership、每个tensor/storage/pinning查询及allocator计数完全不变，
没有跨请求缓存、模型身份猜测、TTL、策略参数或九公式变化。

本次重新读取的官方依据：

- [vLLM CPU开销分析](https://vllm.ai/blog/2024-09-05-perf-update)：同步控制和
  Python对象处理可能影响推理推进；不迁移其旧版本加速百分比。
- [vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  原生加载/激活/LRU仍由核心循环执行；本候选不改这些生命周期。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)：
  pinning是实际storage属性查询，不以普通字段或历史结果替代。
- [Python复制语义](https://docs.python.org/3.12/library/copy.html)与
  [CPython实现](https://raw.githubusercontent.com/python/cpython/v3.12.13/Lib/copy.py)：
  通用深复制逐层处理类型与memo；已知结构的优化仍必须保证可变对象独立。

代码历史核对26fa0a08的观察内复制、2fc3630d/D202的描述投影、D329的跨调用
缓存反例、D352/D663的事务范围，以及D333/D595/D628失败，不重复已否决方案。

## 验证范围与实测

复用D202的probe、NativeHostPhysicalProjection微型真实CPU tensor夹具及
已经保存的D169四张7B状态图，不生成模型/LoRA/负载。动态提取当前提交
`a3cdce75b60d1b55c2c88b9d0d828257349f6b43`的worker方法，仅在实验副本里
替换无staging的复制调用。生产文件SHA仍为
`9fd4f98c9b5241b3b722f7ac5835429c6714c6de5bf1c1f6a2ac77899ff2b270`。

shared、partial view、packed、empty、staged alias、unsupported tensor和
pinning disagreement共7类：全部字段或异常完全一致，实际pinning调用次数
相同；两个返回图之间无共享可变对象。历史四图的来源、容量/allocator解释
和序列化字节数均不变，独立复制与deepcopy逐字段相同。

下表保留全部组件调用块；每块是10次复制的均值，三块交替顺序，仅用于局部
开销观察，不是性能运行重复或CI。没有屏蔽GC，也不猜测个别慢块的原因。

| 旧图中的registered adapters | 通用复制，三块ms | 候选独立复制，三块ms |
|---:|---|---|
| 8 | 9.532 / 9.304 / 9.439 | 24.092 / 2.863 / 2.913 |
| 12 | 35.745 / 36.409 / 14.398 | 4.327 / 4.379 / 4.495 |
| 20 | 25.046 / 45.241 / 45.681 | 29.350 / 7.373 / 7.364 |
| 24 | 50.056 / 29.926 / 53.903 | 30.674 / 30.379 / 30.258 |

存在更慢的块，不能声称每次都更快。全部记录支持进一步验证此局部复制候选，
不证明当前3B的实际复制耗时、清点函数本体已加速、TTFT/TPOT或GPU-s改善。
使用既有Torch2.8 CPU环境，不冒称原生Torch2.13运行；CUDA/原生RPC未执行。

原输入SHA：`ead0c5c81acc52c502874e9d45cfba111665f875220e23467c2e716a0d770602`。
结果SHA：`efae49c951b4ee430fe3ed7267a9126d1661ff6e32cc839b1c786109975cfae7`。
raw为`results/ieee_tc/p2_backend_qualification/d670_20261018/`，保留probe.py、
probe.json/CSV、准入/结果/关闭回执。没有仪器写入生产文件。

资源域3/4GiB、swap0、CPU2,3,26,27；峰值677199872B，high/max/OOM/kill/swap0。
39545已exit0，按身份递归空关闭，GPU无计算作业。147项旧保护未变。

## 决策与下一步

这一个假设进入正式实现与回归阶段，不叠加staged清点合并、跨请求缓存或
其他配置优化。复用worker/footprint/dependency测试，补所有可变字段隔离、
空图、未知schema拒绝、真实调用范围和跨调用变化；再以既有原生环境进行
定向检查。通过后备份，关闭详细观察，按不变3B配置完成新版本一次完整W0。
若完整联合/物理资源无改善，不以局部复制数字替代目标，不重跑同键筛胜。

`system-profile`使本次保持CPU组件范围、原始块和观察边界。变更清单：仅
raw中的probe.py/close.py与本说明/执行状态；没有生产机制修改。完整剩余
3B/基线/dLoRA、同版本7B、A1–A5/S1–S13、成本与审稿/图表任务继续ACTIVE。
