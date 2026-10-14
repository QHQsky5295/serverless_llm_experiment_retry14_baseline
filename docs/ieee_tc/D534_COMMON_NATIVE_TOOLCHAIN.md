# D534：共同编译环境与缓存身份适配

2026-10-15。针对 D532 已有日志直接证明的约102秒 sampling 重编译，修正
实验环境的兼容性，不进行 baseline 性能调参。本文件补充执行身份，不修改
冻结指标 V1／单轮 V2、共同60秒准备窗口或 IEEE 九公式。

## 原因与最小改动

此前两入口共用版本/架构命名的 FlashInfer 目录，但 CUDA 路径表示和 CC
设置不同。Ninja 按变化的编译命令重建；已有共享库并不能保证兼容缓存命中。
证据见 D532_STARTUP_CACHE_REVIEW.md，旧观测及负结果全部保留。

采用同一个显式合同 `configs/ieee_tc/native_toolchain_cuda130_v1.json`：

- 两入口使用已有同一 vLLM 0.30.0 环境、CUDA13.0、GCC/C++、ninja；记录
  真实二进制路径及 SHA。没有安装依赖、升级驱动或替换第三方源码。
- PATH、CC/CXX、CUDA_HOME、NVCC、SM86、MAX_JOBS=2、NVCC线程=1共同
  绑定；未冻结的编译器/链接器附加参数直接拒绝，不静默回退。
- FlashInfer workspace 以合同 SHA 分目录，两个系统共享同一身份的目录，
  不再与旧入口的全局目录互相覆盖。不是每系统一套不等价缓存。
- 父进程在 torch/FlashInfer 导入前绑定；已绑定的子进程必须核对继承值，
  不能悄悄修正变化。PrimeLoRA 输出构建身份回执，vLLM 写入原有工具链回执。
- 仅新 D533 入口和新增 `vllm_7b_resident_native_compiler_v2.json` 启用。
  旧 native v1 JSON、旧性能结果、原工具链目录均不修改。

同一 helper 由两仓库实际入口调用，不复制一个仅在测试使用的环境设置。
新旧 vLLM 配置除 `native_compiler_contract` 外完全相等：RR、四卡DP/TP1、
批处理、LoRA 容量、生成规则及所有 engine_args 未变。PrimeLoRA 未改调度、
层级、准入、模型配置或初始化 profile。

依据：[FlashInfer 编译参数](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/cpp_ext.py)、
[构建和文件锁](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/core.py)、
[Ninja 命令变化检测](https://ninja-build.org/manual.html)。本机实际版本为
0.6.18.post1；在线标签支持接口机制，不冒称整个文件逐字节相同。

## 实际验证与失败保留

16项 CPU 检查通过：8项共享合同检查＋8项现有 vLLM 入口合同检查。
覆盖旧入口不变、父子身份相同、环境/二进制篡改拒绝、未冻结参数拒绝、
缓存命名空间区分、同配置仅增编译身份、隐式把合同带进旧配置被拒绝。

随后分别执行实际 PrimeLoRA import 入口和 vLLM 环境入口，再调用已安装
FlashInfer 原生采样构建文本生成器；两者的回执和4523字节文本完全相等：
SHA `458aaa5a094c7158005abd944e1dc40898f2d7445e550847ccc295c2c6eb9a5b`。
未初始化 CUDA、未调用 build、未构造模型，新 JIT 产物目录没有编译文件。
原共享 build.ninja 和 ninja log 与 D532 审计 SHA 相等。

两次验证脚本问题不隐藏：第一次数组测试加载遇到两个仓库 tests 包名冲突，
8项主仓检查已通过；改为明确加载 baseline 测试文件后，另8项及两个原生
文本探针通过。随后旧文本检查把 configs 中的 fi 误当 shell 结束符；仅在
离线封存时用完整换行结束符检查真实 shell 块，未重复已完成测试或探针。
这些是验证工具错误，不是性能失败，也没有运行 GPU 性能点。

三个资源域均真实递归空、关闭且路径消失；峰值分别1627500544、1572179968、
26677248字节，high/max/OOM/OOMkill/swap均0。147个投稿保护文件及两个仓库
用户修改 SHA 未变。细节在 `paper_results/ieee_tc/main_comparison/d534_native_toolchain/`。

## 共同软件产物边界与下一步

本次只验证绑定，**尚未编译或验证加载新目录里的采样共享库**。下一步在无
推理作业时，以同一合同、受限 CPU/内存、隐藏 GPU，只构建通用 sampling
软件模块；不读取模型权重、adapter、trace 或未来热点，不预热模型/生成。
这是一份双方共同使用的部署前软件产物，和共同环境中的已安装二进制一样
单独记录构建时间、源码/二进制 SHA 与缓存初态，不能称为推理性能改善。

构建后先检查两入口仍得到同一 build.ninja 且 Ninja 无待构建任务，再在新的
执行键运行一次 vLLM W1；D532不能原键重跑取优。随后执行已准备的 D533
Full W1；两次启动前后都封存该缓存身份。模型/adapter 加载、运行内真实
JIT、等待和持卡仍在原60秒通知及生命周期计量内，不作事后扣减。

若后续发现模型相关的新 JIT 工作，计入实际运行，不能只为某一系统提前
跑完整推理热身。通用软件构建不等于消除了全部启动开销；它也不证明 W1
的全部 SLO 退化都由 D532 编译造成。新结果决定结论。

目前不冻结 W1 U_ref/Q_ref，不改旧W0排名、不重跑无受影响证据的全部
baseline、不改共同 warm SLO。独立数值正确性未知的项目仍为未知。
HydraServe D510 隔离路径和4000请求继续复用，不接管其他项目的 k3s。

7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13、离线成本/CE、审稿文档及
最终学术图表仍全部保留；本适配不是系统性能达标或完整实验计划完成。
