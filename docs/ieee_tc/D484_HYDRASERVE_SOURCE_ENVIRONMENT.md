# D484：HydraServe 原生源码依赖环境

2026-10-13。必要兼容适配，不是性能优化或完整复现结果。
主线：HydraServe 实际结果 → 完整可用基线比较 → PrimeLoRA 优化。
现有 k3s 属于其他项目，完全不改动；不以普通 vLLM 替代 HydraServe。

## 依据与处理

固定官方源码 `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b` 的
`scripts/image/Dockerfile`、requirements、setup.py 和 CMakeLists.txt。
Dockerfile 的 Torch2.3.0、xformers0.0.26.post1、CUDA12.4 和 Punica/fastload
保持；已经取得的两份精确 wheel 复用 D482 原文件，不重新下载。
官方提供源码构建路径：[HydraServe](https://github.com/LLMServe/hydraserve)。
本轮网页读取该固定源码失败，具体判断来自本地完整 SHA 固定的官方源码，
不把网页访问失败写成版本已经在线验证。

采用[Python venv](https://docs.python.org/3.10/library/venv.html)创建新的独立
Python3.10环境，不打开 system-site-packages，不修改既有控制/推理环境。
只复用旧环境中 Python-independent 的12份 NVIDIA CUDA运行库：逐包核对
METADATA版本、WHEEL的py3-none标签、RECORD中每个有哈希文件的实际字节，
拒绝越界软链接和Python专属本机扩展。未哈希pyc不作为来源证明。
新环境仅链接 nvidia 命名空间，复制其小型元数据；不把旧Python3.11的
torch/xformers/Triton或其site-packages加入Python3.10的sys.path。
后续禁止pip卸载/升级这些共享链接；容器中旧环境路径应只读挂载。

其余按原common/build requirements安装；CMake限定3.x避免旧Torch的CMake
策略被4.x移除；Ray2.9.3在原>=2.9范围内，Triton2.3.0由Torch2.3.0要求。
datasets2.18.0复用原HydraServe控制环境的版本约束，ModelScope1.15.0对应
官方README。不是根据性能搜索版本，全部解析包与SHA保存pip报告。
原生vllm-nccl专用库、flash-attn2.5.8、fastload及CUDA编译尚须下一步验证；
不能把当前通用依赖通过误写为全部官方后端依赖齐全。

## 执行与安全

脚本 `scripts/ieee_tc_prepare_hydraserve_source_environment.py` 复用D482的
有限pip调用、既有preflight/进程身份及资源观测。独立3/4GiB、swap0、
CPU2,3,26,27，1800秒总上限、16GiB主机内存/100GiB磁盘运行护栏。
新增峰值估计12GiB，启动仍必须满足150GiB/增长门槛。
CUDA不可见，不运行模型、不访问远端工件、不同时构建或执行性能回放。
完成后必须检查pip依赖、实际CPU导入、旧库RECORD未变及所有进程真实退出。

4项新CPU检查通过：正确来源、损坏拒绝、Python ABI拒绝、链接越界拒绝。
初次测试在导入前失败：两个仓库同名scripts包冲突。仅将公共preflight改为
按明确文件位置导入；错误转录、失败scope关闭和修正后测试均保留。
没有通过改原生算法来迎合测试。

## 当前状态与下一步

通用依赖安装已实际完成，耗时360.713657秒。pip check通过，实际CPU导入为
Torch2.3.0+cu121、xformers0.0.26.post1、Triton2.3.0、Transformers4.43.1，
CUDA未初始化。12份复用运行库的RECORD在安装前后均验证通过；未重下载大包。
scope3经InvocationID/inode及实际递归成员为空核查后关闭，峰值3,221,749,760B，
high9008/max0/OOM0/OOMkill0/swap0。high是受限安装时的回收事件，不能写成
所有资源事件为零，也不是推理性能数据。完成时磁盘可用203,557,830,656B。
仍未编译原生扩展，未完成专用NCCL/flash-attn/fastload，不构成GPU资格。
下一步仍是：原生扩展/fastload编译和加载 → D480/D481真实生命周期与私有IPC
接入 → D479真实LoRA层/数值及请求合同 → 100/500-ID/1000/4000资格与结果。
单机四卡是一物理节点，不虚构多节点pipeline触发。

原始目录：实验仓库 `results/ieee_tc/hydraserve_qualification/d484_20261013/`。
新环境：baseline仓库 `installs/hydraserve_native_d484_20261013/`。
工具链仍只读复用 `anaconda3/envs/slinfer_official_20260612/`。
模型、500-ID工件池、既有trace、失败证据、旧投稿结果和用户修改均保留。
