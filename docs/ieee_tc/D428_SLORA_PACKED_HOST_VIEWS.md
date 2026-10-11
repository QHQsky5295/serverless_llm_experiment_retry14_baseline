# D428：S-LoRA 单份 HOST 表示通过等价性验证

2026-10-11，CPU 组件完成；未启动 GPU 或远端服务，尚非 S-LoRA 性能结果。
延续 D425 历史审计、D426 原生计量和 D427 容量证据，未改变实验主线。
本轮采用显式 `packed_views_v1` 表示适配，保留原表示为默认值。

## 问题、依据与改动

D427 已证明当前完整 500 池的原生四副本，仅两份 HOST adapter 张量即需
81.25 GiB，超过共同 80 GiB。顺序启动、修改超时或重复尝试不能解决持久占用。
本轮检查[官方固定版本 InferAdapter](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/infer_adapter.py)：
所用 unified-pool 路径经原生 `load_to_gpu` 读取 `w_combined_home`，分别复制
A/B 到原池，随后 offload；并不消费分模块 HOST 字段的 stride。
该文件的 HTTPS、固定 Git HEAD 和当前本机文件 SHA 完全一致。

采用[PyTorch 2.0.1 视图语义](https://raw.githubusercontent.com/pytorch/pytorch/v2.0.1/docs/source/tensor_view.rst)：
一层只分配一个 pinned packed buffer，按原 A:q/k/v/o、B:q/k/v/o 的顺序
复制当前权重；八个分模块字段改为该 buffer 的视图。packed 的形状、连续性、
stride、数值和后续复制顺序均保持不变。分模块视图可能非连续，因此**不**将
本适配冒充通用 BMM/TP/GQA 加载器；不在不支持的模式下静默回退。

仅支持已有 7B 的 FP16、TP1、非 GQA、真实 swap、非 BMM、原 unified pool。
显式模式检查在 NCCL/CUDA 初始化前拒绝不支持的执行选项；张量/几何检查在
packed HOST 分配前执行。加载失败不发布半成品，不再次初始化覆盖旧层。
3B 等其他路径仍保留原生默认表示，不能自动沿用此适配。

没有改变 paging、batching、调度、admission、prefetch 选择、adapter 获取时机、
GPU swap 路径或 kernel；不合并逻辑 ID，不量化，不利用未来请求或全零权重。
原生加载/换出方法和 InferAdapter 继续使用原源码。IEEE 九个公式未改。

## 验证范围

- 新增 6 项 CPU 测试；原生 Python 3.9.25 的 27 项通过（1.535s）；相关全套
  154 项通过（8.428s）。首次 26 项检查有一个测试夹具误用 FP32 目的缓冲区，
  原生复制要求 FP16，已只修正夹具；失败日志保留，没有改原算法适应测试。
- rank 8/16 的非零小张量逐值比较；一个 packed storage 被全部模块视图共享。
  原生 load/offload 重复、prefetch/普通分支和实际 InferAdapter A/B 复制方法
  在 CPU 模拟设备转移条件下值与调用顺序一致。没有把模拟复制当作 CUDA 验证。
- 使用原池四类独立权重 SHA 各一个静态代表，**全部 32 层**：共 128 层、
  41,943,040 个模块元素；原生和适配后的完整 packed 值、形状/stride、模块值
  均完全一致。重新核验四个代表权重的内容 SHA，没有生成或修改权重。
  该检验通过实际生成的 source-view loader，不是仅调用一段独立替代计算。
- 实际 Python3.9.25/PyTorch2.0.1+cu118 导入该 source view 的 RPC、API、loader
  通过，CUDA 未初始化。只复制三个源码文件，其余链接原始源码；原嵌套 checkout
  和十文件历史兼容补丁未修改。
- pinning 在 CPU 检查中显式模拟；没有测实际 pinned RSS/完整部署峰值，也没有
  宣称通过 GPU 数值资格、500 个逻辑 ID 执行或 4,000 请求比较。

四副本 adapter 数据的表示下界由 81.25 GiB 变为 **40.625 GiB**；这是表示大小，
不是已测服务 RSS 或 50% 生命周期 GPU-s 改善。backbone staging、allocator、
运行时、page cache 等继续计入服务资源，后续实际 gate 仍不可省。

## 接口和证据

复用 baseline 的既有 source-adapter：

```text
prepare_ieee_tc_serverless_stack.py prepare-slora-measurement-view
  --native-source <已有 S-LoRA checkout>
  --main-repo /home/qhq/serverless_llm_experiment_retry14_baseline
  --output <新的独占目录>
  --host-layout packed_views_v1
```

默认 `native_v1` 不变。manifest 显式记录表示变化、适用范围、15 个原文件 SHA、
三个派生文件 SHA、helper SHA；`performance_run_authorized=false` 不变。
下界计算也显式选择表示，不将“未超必要下界”自动判成部署合格。

Raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d428_20261011/`。

- source-view manifest：`f19bae720380c4dfce2f1a0e916c6b03a649faf1530e41d0316ddad68a78792b`
- 真实权重等价证据：`fb21a4f98761d5361a37e9644515333ce7c2a18ef3db9d7dbc425f10eec6ba12`
- helper：`88f73314cb1421d93d67d08ce436ccb3846e259edce1c6ced506404aac24f7ce`
- source adapter：`5bd12684c78129d796bceada473e1f9300d1390327d323c92a6987f9661e25ce`
- 154 项回归日志：`fa48a47923beb49904abeefb2ab0d4047447cd680a475c4b93a1ff6094144197`

Curated 证据包 `paper_results/ieee_tc/baseline_audit/d428_slora_packed_host_views.tar.gz`
为 20 个普通文件、22,905 B，逐成员与 raw 字节一致；SHA
`7494da840f1c558c9e224414ac29489dd008c639141f66630c0f75f638d85f8f`。
包保留失败测试日志、成功测试、真实权重检验 recipe/结果、三个派生源码、
manifest、上游来源回执和资源收尾日志。15 个原文件、三个派生文件及两个
helper/source-adapter SHA 均核验。147 个历史保护项零变化。
六个本轮 CPU scope 顺序执行、核对身份及全部进程退出后关闭；high/max/OOM/
swap 均为零。最新 GPU compute census 为空；磁盘可用 163,378,855,936 B。

Baseline 组件提交 `f51ffd7f7d278bef1a5ac690dd1aae0b97963620` 已推送
`origin/main` 并核验远端 SHA；这是 CPU 组件备份，不是性能运行的执行版本。

## 下一步：回到共同部署接入

不再扩展局部内存调优或重复上述 CPU 检验。接入既有 pre-ready open-loop、
S-LoRA 显式 native validator、原生实际 worker 身份/ready 和物理 GPU leases，
在共同 60 秒部署通知后完成原生全池 HOST 准备并计量；不平移业务时刻。
完整进程限制、监控和真实释放合格后，执行受影响 100 请求、缺失 500-ID/
1,000 资格及一次完整 W0。CPU 表示等价不能代替这些必要证据。

PrimeLoRA/ServerlessLLM/vLLM 已封存 W0 不重跑。其余 7B baseline、W1/W2/G2、
3B、消融/机制/敏感性和文档交付仍 OPEN。本轮没有图、清理、远程缓存重发布、
环境安装或论文修改。
