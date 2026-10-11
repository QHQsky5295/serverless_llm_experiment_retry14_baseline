# D428：显式 S-LoRA packed HOST 适配

2026-10-11，CPU 等价性通过，GPU/服务资格待完成。

针对 D427 确认的四副本两份 HOST 权重 81.25 GiB 超限，既有
`prepare_ieee_tc_serverless_stack.py` 增加显式 `--host-layout packed_views_v1`。
默认原生表示不变；新的独占 source view 仅额外替换层级 CPU loader：一份
pinned packed buffer＋八个值相同的模块视图。原生 GPU 换入/换出、池复制、
kernel、调度和预取选择不变。表示变化写入 manifest，不冒称纯计量修改。

限定 FP16、TP1、非 GQA、真实 swap、非 BMM、原 unified pool。分模块视图的
stride 可能改变，因此不能用于 BMM；RPC 在 NCCL/CUDA 初始化前检查执行模式。
其他不支持几何和张量在 HOST 分配前拒绝，无 fallback。原嵌套源码不改。

原生 Python3.9 27 项通过 1.535s，完整 154 项通过 8.428s。首次测试夹具的
FP32/FP16 不一致失败日志保留，只修正夹具。两种 rank 的非零 fixture、原生
load/offload、实际池 A/B 复制方法在 CPU 模拟设备转移下等价。
原池四类权重代表的全部 32 层也通过实际 source view 逐值比较：128 层、
41,943,040 个模块元素。四代表 SHA 重新核验；没有新建权重/负载。
Python3.9.25/torch2.0.1+cu118 实际导入 RPC/API/loader 成功，CUDA 未初始化。

只证明 HOST 表示等价，不是 pinned RSS、CUDA 性能或 500 逻辑 ID 执行验证。
adapter 数据四副本下界变为 40.625 GiB，完整服务峰值仍待量测。

主仓库详细报告：`docs/ieee_tc/D428_SLORA_PACKED_HOST_VIEWS.md`。
Raw：主实验 results 的 `ieee_tc/slora_qualification/d428_20261011/`。
source-view manifest SHA `f19bae720380c4dfce2f1a0e916c6b03a649faf1530e41d0316ddad68a78792b`；
真实权重证据 SHA `fb21a4f98761d5361a37e9644515333ce7c2a18ef3db9d7dbc425f10eec6ba12`。
主仓库 20 成员证据包（22,905 B）SHA
`7494da840f1c558c9e224414ac29489dd008c639141f66630c0f75f638d85f8f`，逐成员等于 raw。
147 个历史保护项零变化，六个本轮 CPU scope 已安全关闭，无 OOM 或 swap。

依据：[S-LoRA 原生池复制](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/infer_adapter.py)、
[PyTorch2.0.1 视图](https://raw.githubusercontent.com/pytorch/pytorch/v2.0.1/docs/source/tensor_view.rst)。
前者 HTTPS 与 Git HEAD、当前源码逐字一致；后者明确 view 共享 storage 与
非连续性边界。本机适用性由实际旧版 PyTorch 的测试证明，不凭文档代替执行。

下一步停止局部扩展，接入现有共同启动/计时/physical-owner；原生全池 HOST
准备保留且计入共同通知后的生命周期。必要资格及一次完整 W0 仍待执行。
用户 dirty replay/relayserve、旧性能结果、其他三系统封存点均不改不重跑。
