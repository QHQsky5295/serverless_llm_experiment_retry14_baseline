# D439：Loquetier 在线请求边界与 KV 生命周期检查

日期：2026-10-12。状态：必要适配和 CPU 检查通过；尚未接通实际 HTTP
worker、真实远程准备与共同结果分析，**不是新的 GPU 性能实验或完整复现成功**。
完整计划、冻结指标 V1、单轮 V2 和最新基线执行规则继续生效。

## 依据与不变部分

接续 D437/D438。D438 的单卡原生加载及 KV 容量检查已通过，不重复运行。
其执行 commit 和旧 helper SHA 保留；本轮公共 helper 的新字节不能冒充旧身份。
核查了先前修改记录以及[Loquetier 官方仓库](https://github.com/NJUDeepEngine/Loquetier)。
具体调用顺序依据本地固定 HEAD `aae33baeeb19777129c1ccbff99a898d4a0e2c63`
的 `model_workflow.run_inference` 和 `model_generation.ModelGeneration`；
之前固定 SHA 的 raw URL 读取失败，不声称网络复核了源码全部字节。

官方工作流按到达顺序添加请求，在达到 batch 上限或等待下一到达时执行原生
`generate`。现在需要将这一路径接入共同的在线开放回放，不能预读未来 trace，
不能把原离线入口的 ready 时间当作共同业务开始时间。

本轮只新增必要的在线输入、取消、终态和资源释放适配。保留虚拟模块、SMLM、
原生批处理和 LoRA 排序、精度与 D437 四个静态分片；历史 batch=2 是既有
兼容工作点，不称官方默认或最优。没有基线性能调参或配置搜索。

## 新增边界

复用现有 `replay_vllm_engine_trace.py`，未增加独立实验框架：

- `LoquetierStaticRouter` 只按冻结 adapter→worker 映射选择分片；等待该分片
  ready，不改投已经 ready 的错误分片，不引入动态放置、队列感知或 PrimeLoRA 策略。
- `LoquetierOnlineRequests` 接收真实在线到达，FIFO 填充至既有 batch 上限；
  原生初始化和所有状态操作归一个线程所有，GPU 迭代不阻塞接收新请求。
- `LoquetierNativeBatchBridge` 调用原生 `add_requests/generate`，要求完整静态
  registry。输入为确切 CPU token IDs，固定 greedy 目标 1–256，上下文≤1024；
  未知分片、重复请求、超限输入直接失败，不默认选择 adapter 0。
- 终态只来自原生 finished mark；完成数量、逐 forward 数量、顺序时间、实际
  process identity 全部核对后仅消费一次。保留真实输入/输出 IDs 及 SHA。
- 取消只在原生迭代边界处理；未登记请求不分配 KV，已登记请求只释放自己的
  原生 KV，其他请求保持原顺序。部分输出作为取消证据，不算成功工作。
- 取消与原生终态同时发生时保留原生完成记录并标 `client_cancelled=true`，
  不向已经取消的请求返回成功。关闭服务先等当前迭代结束，不并发改动 KV。
- 初始化、登记或原生计算失败后不重试、不更换请求、不回退另一后端。
  后续 HTTP worker 必须把此状态作为致命服务失败，退出并由既有生命周期
  控制收尾；特别是原生分配中途抛错，不能仅凭队列清空声称所有 GPU 资源释放。

计量仍是原有整数 `.item()` 后的 CPU 可见时间，不冒称 kernel 内部时间。
结构绑定、远端内容和模型数值正确性分开；`lora_numerical_correctness_qualified`
保持 false。新 schema 为 `ieee_tc_loquetier_native_v1`，不假装成 vLLM 或 S-LoRA。

## 发现的 KV 生命周期问题与最小修正

本地固定源码中，`post_process` 在最后请求结束时释放 KV 并清空 requests，
同时将 `decode_request_changed` 置 false，却保留 `cached_decode_kv` 的引用。
下一次新 prefill 到来时，`pre_process` 会遍历该旧列表并执行 `acquire_one()`，
使已经结束的请求重新分配 KV 页。这不是推测：测试执行实际原生
`add_requests/pre_process/post_process` 和 CPU `KvPool/KvCache` 重现了该路径。

最小见证：首请求结束后 pool 为 0 页；添加第二请求并预处理后，原行为保留
首请求新分配的 1 页，总计 2 页。仅在批次为空且旧 cache 的 `num_pages` 和
`seqlen` 均为 0 时清除旧引用，第二请求总计 1 页，首请求仍为 0 页。
对仍有活跃请求的批次不操作；空批次残留活页则显式失败，不能静默丢弃。

此修正放在原生 `generate` 后的在线适配层，原 checkout 不改动；整体被测
行为仍须披露这项生命周期修正，不能称完全未经适配的官方二进制。
它不是新 KV 策略，没有改变活跃批次、计算、精度或缓存容量；未测量其性能收益。

## 验证与保护

- 现有 `loquetier_20260520` 环境运行 26 项测试通过（2.317 秒，无 skip）：
  源码视图、原生整数计量、实际原生 KV 生命周期、静态映射、异步 FIFO、
  取消/终态竞争、初始化/登记/计算失败和关闭边界。
  模型算术由明确标记的 CPU 小张量或 synthetic bridge 替代，不是推理性能证据。
- 既有 ServerlessLLM、vLLM、S-LoRA 公共回归 163 项通过（19.134 秒）。
  负面断流测试的连接关闭日志属于预期；没有更改这些系统的策略或历史结果。
- GPU 隐藏；CPU 2,3,26,27；辅助资源域 high=3/max=4 GiB，swap=0。
  最终 Loquetier 测试 peak=254386176 B，公共回归 peak=800219136 B，
  保护核查 peak=29315072 B；high/max/OOM 均为 0。
- 旧 147 个保护条目核验零变化，三个用户修改文件 SHA、Loquetier 三个历史
  补丁和 S-LoRA 十个历史补丁 diff SHA 保持不变。辅助进程树真实为空后，
  按对应 InvocationID 关闭，不处理无关进程。
- 无 GPU 模型加载、远端管理、数据下载、重新发布、工件池复制、历史覆盖或绘图。
  用户再次批准一次性只读交付缓存；继续复用 D78/D80，不重复制作。

本轮改动与资源身份见 `paper_results/ieee_tc/loquetier/d439_cpu_boundary_summary.json`。
此前两个开发检查分别为 21/23 项通过，不是性能重复或正式均值。

## 下一主线和未完成范围

继续接入已有公共 HTTP/真实远端 resolver/物理资源租约，增加 Loquetier 原生
worker、启动身份、错误退出和显式分析合同；必须创建新的源码视图 manifest，
不能沿用指向已变化 helper 的 D438 manifest。再进行既定 100 请求、500-ID
完整功能覆盖、原 1000 请求兼容、单轮完整 4000 请求回放。当前没有 Loquetier W0 行。

HydraServe、其余 7B 比较、必要 PrimeLoRA 优化、W1/W2/G2、3B、消融、机制、
敏感性、审稿文档与后续图表全部仍待完成。此 checkpoint 不缩减计划或关闭总目标。
