# D674：主动 HOST→GPU 准备中的事务内重复清点

2026-10-18。D673 承诺的有界调用审计完成；本轮没有修改生产代码、启动 GPU、
重跑 D668/D672 或改变配置。3B 最新完整观测仍为 D672：联合 3647/4000
（91.175%），U=14931.314879785059 GPU-s；仍差153条联合达标，独立数值
证明仍未知，不能宣布 G1/G2 闭口。

## 1. 复用证据与原始数据

复用 D668 的1000请求前缀诊断：四个原生进程、9233条完整 HOST 清点记录，
以及其已封存的统计函数。按 `(process,parent)` 分组，再按真实开始时间排序；
不把不同进程或不同事务的相邻调用混为一次。共得到32类操作/状态/清点序列。
D674 的补充分析只区分清点次序，不新增剖析或请求。

`ieee_prepare_host` 共1425次父调用；其中982次具有三个完整清点，其他443次
没有完整清点。如下计数仅属 D668，不能外推为 D672 全4000请求的发生次数。

| 三清点序列 | 事务数 | 首次与第二次范围/数量/字节相同 | 第二次与第三次相同 |
|---|---:|---:|---:|
| registered → registered → registered | 942 | 855 | 942 |
| registered → registered+staged → registered+staged | 40 | 0 | 40 |
| 合计 | 982 | 855 | 982 |

相同范围名称并不足够：第一行中仍有87次在前两次之间改变了 adapter/存储
数量或字节。只按名称合并就会漏掉真实变更。另40次含 staged 范围差异。
这个127次反例直接排除了“整次主动准备只取第一份图”的无条件方案。

在942次全 registered 范围调用中：

| 父调用入口状态 | 事务数 | 第三次清点均值，ms | 第三次清点累计，ms |
|---|---:|---:|---:|
| 入口已有未完成客户端工作 | 157 | 67.521211 | 10600.830188 |
| 入口没有上述工作 | 785 | 69.366343 | 54452.579147 |

这些是原生主机函数时间，不是 CUDA kernel 时间、请求新增延迟或保证可删除的
时间；同一进程内外的活动可能交叠。入口状态也不是每个子调用时的状态。
不把累计值从 TTFT/TPOT 中相减，不据此估计达标请求数。

## 2. 调用链与历史边界

当前实际路径为：

1. `_core_prepare_host` 在 UniProc owner 中取得同一次 scheduler/KV/length 观察。
2. `proactive_host_prepare_and_acquire` 调用 replacement cost provider：第一次
   完整 registered HOST 图，计算原有 GPU/HOST fallback loss。
3. 必要时移除 HOST victim、把 staged 对象注册到 CPU cache，然后执行已决定
   的 GPU victim 处理。这一边界可能改变 HOST 成员和物理图，不能复用第1份。
4. `_load_and_acquire` 的预算前检查：第二次完整 registered/staged HOST 图，
   读取 allocator、执行原有预算/workspace 检查。
5. `preparation_loader` 通过现有 dense pitched-copy 将已注册 HOST 权重复制
   至 GPU，保持原生 setter/resetter、fence、acquire/pin。
6. 预算后检查：第三次完整 HOST 图和新的 allocator/workspace 检查。

D352 已减少 confirmed GPU-hit 的后清点；D663 只处理具有匹配 source
incarnation 与活跃 demand-preparation HOST lease 的请求驱动激活。
主动路径没有这一 demand lease，也未传入该 source identity，因而仍进入完整
后清点。D671 仅改 source_snapshot 的独立图复制，不涉及这里的三清点。
D355 已做控制事务之间的模型步交错；本轮不拆开一个原子事务以绕过其约束。

D329 的六类反例继续有效：同 epoch/model ID 并不保证 tensor、storage、
alias 或 pin 不变。第2、3次记录一致只是定位线索，不是省略检查的安全证明。
冷文件加载、staging 注册和不同 RPC 之间仍须新鲜完整观察。

## 3. 当前官方实现依据

本轮重新读取官方版本和本机实际安装路径；相关本地文件 SHA 已入 source_refs，
不声称整个安装目录与远端逐字相等。

- [vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  已注册且 non-inplace 的分支触碰现有 CPU 对象，再激活 GPU；不同于加载/注册分支。
- [vLLM 0.30 model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)：
  激活使用已注册对象和原生逐层 setter；GPU LRU 回收与 CPU 对象移除是不同路径。
- [UniProc executor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)：
  同步 collective RPC 直接执行 worker 方法。当前桥接限定该 owner，不推广到远程多进程执行。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)：
  pinning 查询涉及底层存储。后台 allocator 状态仍可能变化，所以后检查必须重新读计数。

官方路径支持进一步核查“已注册对象只读激活”的假设，不提供本机加速比例，
也不自动证明本地 pitched-copy 扩展正确。该扩展已有严格 source/destination
匹配、实际 CUDA copy 和 finally fence；下一步还须验证其与存储事实复用的组合。

## 4. 唯一下一候选与验证边界

候选：只对**已完成必要注册、当前已注册且源身份匹配、由已审计主动 loader
同步激活**的事务，把第3次图遍历替换为第2次图的事务内存储事实；继续取得
新的 allocator/workspace 结果。不是使用第1次图，不是跨请求缓存。

实施前/最小验证必须证明：

- 权威 source incarnation、owner、实际 loader/manager 方法绑定和未改变的
  dense TP=1 复制合同同时成立；仅一个布尔开关或 Python 对象 ID 不算证明。
- 验证区间从第2次新鲜完整图之后开始；不能跨越任何 CPU registration、
  removal、packing、dtype/布局转换或 owner yield。此前发生的变更必须已经
  纳入第2次图；未能证明的路径保留完整检查，不能吞异常后降级为成功。
- 保持实际 GPU copy、victim/loss 判定、Eq.(8)、fence 和引用生命周期；新的
  allocator 增长仍能拒绝，失败仍使 owner 失效，不伪造 rollback。
- 回执把主动路径与 D663 请求驱动路径分开。既有 GPU-hit、无 source identity、
  staged/冷加载、异常、取消、transport replay、每事务重新测量等回归不放宽。
- 先复用现有 tiny CPU fixture 和原生资格入口验证触发域，再进行一次无详细
  observer 的新版本完整3B W0；保持工件、trace、模型配置、SLO和计费合同。
  不重复同键寻优；组件少一次调用不等于完整服务收益。

本轮只选定这一待验证假设，未实现、未接受性能收益。暂不同时合并 cost
provider 与预算前检查，不加入 file-host reuse 等另一优化，不重开宽泛 profiling。
若资格失败，按失败证据修正或拒绝；若完整回放未改善，保留结果并回到已有
控制/路由等待证据，不能继续堆叠未经验证改动。

## 5. 产物、安全与主线

原始：`results/ieee_tc/p2_backend_qualification/d674_20261018/`。
`audit.py` 直接复用 D668 stats；`audit2.py` 新增次序签名核对，保留第一次
分析。发布前检查拦截了一条来源索引错误：audit2展开时复用变量，把D668
脚本路径与本轮audit.py的SHA误配。原两次结果、失败发布身份和关闭回执
全部保留；只修正这一个来源配对，所有计数、测量和结论不变，23项来源SHA
重新通过。最终 `diagnosis3.json` SHA：
`0d9d3524f0d280600c24ed7d177b87131d7516d8e58387e32bf2f72e2b5fea11`。
curated：`paper_results/ieee_tc/p2_backend/d674_proactive_host_inventory_audit/`。

两个CPU分析域都成功、递归空并关闭，峰值43610112/43319296字节，
high/max/OOM/kill/swap均0；147旧投稿保护通过。没有新的 GPU 性能实验、
远端服务操作、依赖安装、工件或负载生成，也没有触碰非项目 k3s。

按 system-profile/analyze-results 区分实际测量、解释和待证假设；其文件变更
只有本轮 audit/audit2/close/publication 小型管理脚本、本文件与执行状态及
curated 证据，无生产 instrumentation。github-sync 在提交前核对明确路径、
旧保护/用户SHA、源码/归档SHA、语法和秘密扫描；晚D673发布回执随本轮保存。

3B W1/W2、3B基线/dLoRA、同版本7B主结果与Full、A1–A5/S1–S13、成本/CE、
两轮审稿文档和后期图表全部继续ACTIVE。当前不提前进入这些阶段，不画图，
不把本次有界诊断当作新的达标实验。
