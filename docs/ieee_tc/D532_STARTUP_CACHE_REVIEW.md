# D532：启动异常中的共享编译缓存失配

2026-10-15。有界离线核查；不是新性能运行或独立审计。采用 analyze-results
的观测—解释—证据边界流程。D532 全部原始结果、2655/4000 联合达标及
9007.169697 GPU-s 均保持不变，不删首阶段，不从请求时延减去编译耗时。

## 观测

| 项目 | D423 W0 | D532 W1 |
|---|---:|---:|
| 四个原生 worker 就绪，距共同部署通知 |40.887–41.793 s|142.934–143.351 s|
| worker 0 引擎初始化日志 |9.59 s|111.57 s|
| 模型 AOT 缓存 |命中|命中|
| 启动前采样 build.ninja 字节数 |3731|3778|
| 本次采样重编译记录 |不由此表推断|101.758 s 构建终点|

两场景到达过程不同，本表仅定位请求前初始化差异，不将 W0/W1 性能差当作
单因素因果对照。四 worker 的 engine_args 和已记录工具链相等；旧/新 runner
的 native_worker_environment、_native_worker、NativeVLLMRequestHandler 三个
实际函数 AST 相同。没有证明整个历史环境及所有安装文件逐字节相同。

启动前两个缓存清单各628文件；只有 FlashInfer sampling 目录下七个文件
发生变化，模型 AOT 文件未变。本次 `.ninja_log` 的前3780、4320字节分别
精确匹配 D423、D532 启动前封存 SHA，新增四条构建记录。因此这里不是
仅凭文件修改时间猜测发生编译，而是由封存前缀和追加记录建立来源关系。

新增构建记录包括三个对象与一个共享库：最早开始9 ms，链接完成101758 ms，
覆盖101749 ms；构建日志终点约为本地00:35:19，与原生日志00:33:37至
00:35:21的空档吻合。该区间100个一秒监控采样中96个四卡 utilization 均为0；
这是辅助观测，不是逐 kernel 测量，也不能独立证明每个 worker 的等待原因。

## 为什么已有缓存仍重新编译

实际两套入口使用同一目录：
`~/.cache/flashinfer/0.6.18.post1/86/cached_ops/sampling`，但构建描述不同：

| 字段 | 独立 vLLM 入口 | PrimeLoRA 入口 |
|---|---|---|
| CUDA 根目录 |`/usr/local/cuda`|`/usr/local/cuda-13.0`|
| NVCC 命令 |`$cuda_home/bin/nvcc`|显式 `/usr/local/cuda-13.0/bin/nvcc`|
| NVCC host compiler |未显式设置|`-ccbin /usr/bin/gcc`|

两个 NVCC 路径解析到同一实际文件。PrimeLoRA 主 runner 在 CC 未设置时选择
`/usr/bin/gcc`；这一初始化块在保留的 D476 源码中已经存在，不是本轮新改动。

第一项 CPU 探针仅复现两个 CUDA 环境变量，未匹配历史3778字节描述；此负
结果完整保留。随后按真实 runner 补入 CC，调用原生生成器只在内存生成文本，
得到3778字节与历史 SHA **053d80c8059d08ac41c81a4dc43437825afeed6c60524dc3fe0cc588500ac07f**
精确一致；默认入口生成3731字节，与 D423 及 D532 构建后的 SHA
**98c0b5e1fdebe080604bdf3b08bf39d9ed05c61e35e06037e0756d4d561756d1** 一致。
两探针隐藏 GPU、未构造模型、未调用 build，也未修改共享缓存。

原生 FlashInfer 会在同一个模块路径重写变化的构建描述，并由 Ninja 判断
是否需要重建；Ninja 将命令变化作为重建依据。该机制与本次实际追加构建
记录一致。它解释了为何切换实验入口会使“磁盘上存在缓存”不等于缓存可复用。
来源：[FlashInfer 构建与锁](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/core.py)、
[缓存目录规则](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/env.py)、
[Ninja 手册](https://ninja-build.org/manual.html)。本机版本0.6.18.post1；引用支持
设计与调用约定，不宣称它与0.6.18整个文件完全相同。

vLLM V2 的真实调度式 warmup 在图捕获前执行采样，因而模型 AOT 命中不能
排除采样模块重编译。[官方 warmup 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu/warmup.py)、
[调用顺序](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_worker.py)。

## 裁决和下一步

1. 共同运行环境的编译描述失配已获直接证据；不能将这约102秒的主要启动
   差异归为真实远端网络、adapter 层级机制、vLLM 调度缺陷或模型 AOT miss。
2. D532 仍是实际完成、计量有效的观测；不因不达标删除或改数字。但目前不
   用它冻结“缓存初态可比”的 W1 参考，也不据此宣告 PrimeLoRA 机制领先。
3. 还没有逐函数 warmup span、逐 worker 锁等待或完整历史环境。因此不声称
   精确解释全部104秒空档，更不把全部 W1 达标率退化都归给重新编译。
4. 下一步只修正共同执行环境：显式绑定并核验两入口的真实编译工具身份，
   建立编译缓存的兼容/初态规则，避免不同描述互相覆盖。优先复用既有启动、
   缓存封存与测试入口，不改变调度、并发、模型参数、60秒通知或共同 SLO。
   不允许仅对 vLLM 偷偷预热；若采用发布前的共同软件构建，必须明确与模型、
   adapter、GPU 预加载区分、冻结新执行身份及影响范围。真实运行内编译仍计量。
5. 只有此必要适配通过受影响验证后，才继续新的 W1 参考与已准备的 D533。
   不能原键重复取好结果，也不因这里的环境问题重跑全部历史 baseline。

证据目录：`paper_results/ieee_tc/main_comparison/d532_vllm_w1/startup_review1/`。
review.json、compiler_binding.json 保留完整匹配/不匹配探针与来源；两个 CPU
资源域已实际清空关闭，峰值分别387276800/383705088字节，high/max/OOM/swap0。
无 GPU、远端维护、安装、构建或编译缓存清理；共享 k3s 未动。

本核查不关闭7B W1/W2/G2、3B全部系统、dLoRA B6、A1–A5、S1–S13、成本/
CE、两轮审稿文档和最终图表。HydraServe D510 隔离适配及全4000结果继续复用。
