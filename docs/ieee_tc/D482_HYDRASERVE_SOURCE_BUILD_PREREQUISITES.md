# D482：镜像获取失败保留，转入官方源码构建依赖核验

2026-10-13。此记录是必要环境适配，不是性能结果或基线调参。
保持执行顺序：HydraServe 实际资格/结果 → 完整可用基线比较 → PrimeLoRA 优化。

## 已确认的失败与范围

D479 镜像续传在2479.589秒结束：数据流TimeoutError后，重新认证遇到
URLError/SSLEOFError(errno8)。认证位于原数据流重试的except分支之外，
因此只记录一次流失败就退出。不是OOM，也没有启动模型。
小blob保留119,634,955/189,594,758B，大blob903,052,834/3,626,807,089B；
64个已验证完整blob、两个partial inode和官方manifest身份均未改变。
实际原进程不存在，递归cgroup成员为空，再按InvocationID/inode关闭；
峰值81,137,664B，所有high/max/OOM/swap为0。旧失败不覆盖。

D482复用D479获取器，只将认证、prefix/interior核验和range数据流纳入
同一每blob最多8次失败预算。保留HTTPS身份验证、精确206/Content-Range/
Content-Length、inode/offset、完整length/SHA及原10800秒/100GiB/16GiB护栏。
拒绝证书错误、403、错误字节和安全门槛的无效重试；不打印token/代理值。
7项CPU回归通过。依据[Registry V2协议](https://distribution.github.io/distribution/spec/api/)
保留partial并使用Range续传；最终digest仍必须与固定manifest一致。

这次有限续传仍因8次相同TLS错误在122.163秒结束，没有新增字节，也没有
导入镜像。实际scope已关闭，峰值28,704,768B，所有内存事件/swap为0。
不能把反复失败的软件下载描述为仍在推进，不能凭这个判定HydraServe慢。

只读连接检查：curl8.12.1经当前代理访问registry/auth均status000、exit35；
TLS1.2同样失败。直接连接两端均超时exit28。相同代理访问GitHub API为200。
实际下载进程8个proxy环境项与启动环境逐项相等（仅记录相等性，不保存值）。
这些证据定位到镜像获取通路，尚未证明具体网络管理策略、远端故障或根因。
此Internet软件仓库与实验真实远端LoRA的1Gbps链路不是同一个条件。

## 安全替代：相同官方代码的源码构建

[官方README](https://github.com/LLMServe/hydraserve)明确提供源码构建路径。
仍固定commit `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`，不换成普通vLLM。
Dockerfile使用CUDA12.4.1，依赖固定Torch2.3.0/xformers0.0.26.post1；
原生fastload、Punica、pipeline、启动重叠和consolidation仍须保留及资格。

本轮只读核验发现此前“仅有宿主CUDA13.0”不足以排除本地构建：
`/home/qhq/anaconda3/envs/slinfer_official_20260612`内已有nvcc12.4.131、
ptxas、CUDA头文件、libdevice和cudart12.4.127。本轮实际执行nvcc --version
成功，但尚未证明完整链接/内核兼容性。只允许只读复用此工具链，不改旧环境。
该环境Torch为2.3.1，不能拿它冒充官方要求的2.3.0。其11项已安装CUDA
运行库版本与Torch2.3.0公开精确依赖相符，可进一步核查只读复用；Triton、
Python ABI、fastload、flash-attn和NCCL选择仍需完整验证。

实际取得PyPI元数据中的cp310/Linux x86-64精确软件包身份：

| 包 | 字节 | SHA256 |
|---|---:|---|
| torch2.3.0 | 779141888 | d8ea5a465dbfd8501f33c937d1f693176c9aef9d1c1b0ca1d44ed7b0a18c52ac |
| xformers0.0.26.post1 | 222745400 | 1fe0f9a3dddb7f6175c2e34de2f318d6742eec28c068b8334b093016b2c9c2c8 |

使用既有Python3.10/pip只下载这两个指定包，关闭额外pip缓存，不安装到旧
环境，不下载模型或LoRA。下载单独3/4GiB、swap0、CPU2,3,26,27，启动时
150GiB/增长门槛已通过，运行检查100GiB/16GiB与1800秒上限。完成必须核验
字节与PyPI SHA，再关闭实际所有进程。此文记录依赖获取启动，不预写成功。

## 下一步与证据边界

完成依赖获取后，核查磁盘门槛；必要时只无损压缩已关闭证据并双重SHA核验。
在全新隔离环境中复用已核验工具链/运行库，构建固定官方源码和fastload；
最多两个编译任务，不与性能实验并行，不升级驱动或修改已有环境。
然后将实际后端接入D480生命周期、D481私有IPC/地址与D479 LoRA绑定，
验证原生GPU计算/数值/身份/固定输出/计量，依序100→500-ID→1000→4000。
准备工作不能代替真实资格或性能，不伪造单机为多物理节点，不改其他项目k3s。

原始目录：`results/ieee_tc/hydraserve_qualification/d482_20261013/`（实验仓库）。
curated目录：本仓库`paper_results/ieee_tc/hydraserve/d482/`。
保留D479/D482全部失败，运行中下载日志和大包不进Git。完整后续实验范围不变。
