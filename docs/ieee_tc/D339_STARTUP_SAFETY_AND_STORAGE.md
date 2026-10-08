# D339：启动候选的安全性检查与无损空间回收

2026-10-08。主线仍为 7B 优化；本轮没有 GPU 性能实验，没有新增图表。

## 结果与裁决

在独立 CPU 资源域中，试验性 native-fork 路径的 170 项单元测试通过，
但真实 vLLM 0.30.0 导入后的前提检查未通过：CUDA 未初始化，Python
线程为 MainThread 和两个 `humming-bg-build-reaper`，操作系统任务数为 4。
因此**不启用 fork、不启动该候选的 GPU 回放**。这不是证明 fork 必然失败，
而是当前依赖环境不满足本次预登记的无额外线程前提。试验补丁已归档后撤销，
生产启动仍为独立 frontend + native spawn，未放宽安全检查。

历史 D337 定位到约 47 秒的 activation；D338 已否定“删除 sitecustomize
即可省掉其全部导入时间”的无依据解释。D339 不重复这两个实验。
[vLLM 官方说明](https://docs.vllm.ai/en/v0.30.0/design/multiprocessing/)
明确提示 fork 与线程依赖的风险；该文的历史代码说明写于 2024 年，实际行为
同时核对了本地 0.30.0 的 system_utils、core_client 和 engine/utils。

进一步读到 Humming 0.1.12 在模块导入时启动两个后台构建子进程；各自用
daemon reaper 线程等待，stdout/stderr 被组件重定向。当前不把它们的耗时
推造为 TTFT 损失，也不声称发生了编译或解释了全部启动开销。
后续唯一候选 D340：在实际未量化 FP16 路径中，使用其原生
`HUMMING_DISABLE_PARALLEL_BUILD=1` 将可选构建保留为按需执行；**保留 spawn**。
参考[官方 Humming NVRTC 实现](https://github.com/vllm-project/humming/blob/main/humming/utils/nvrtc.py)
和本地安装源代码，具体安装源 SHA 随证据保存。不改安装环境源码。

## 空间与资源

对三个已结束、已确认释放 GPU 的失败运行的大型 `main_outcome.json`
复用既有无损压缩程序：D99 3B attempt6、D133 7B full1、D98 3B attempt5。
各自先与历史 curated SHA 核对，gzip 检查、两次解压 SHA 和再次原文件 SHA
通过后，仅移除未压缩重复件。失败身份、内容、唯一工件/权重/trace 全部保留。
总回收分配空间 **4,213,305,344 B**；完成时可用 **164,836,679,680 B**。

压缩峰值 3,221,626,880 B，memory.high 事件 3599，max/OOM/swap 均为 0。
这发生在非性能测量期间，不作服务内存结果。CPU 检查峰值 2,369,814,528 B，
high/max/OOM/swap 为 0。所有 D339 scope 已验证为空并按精确身份关闭。
首次 CPU 启动用了不合规范的 scope 名称，入口在导入前拒绝；失败和关闭记录保留。
147 项保护清单仍只有此前已登记的 baseline replay 脚本差异，不能称全部零差异。

## 剖析修改记录

| 文件 | 用途 | 最终状态 |
|---|---|---|
| raw D339 `test_candidate.py` | 单元测试及真实 CPU 导入安全观察 | 保留 |
| raw D339 `rejected_fork_candidate.patch` | 四个文件的试验代码与测试 | 仅归档，生产撤销 |
| raw D339 `compress_historical.py` | 复用既有无损压缩与引用检查 | 保留 |
| raw D339 `close_compression.py`, `close_cpu.py` | 精确资源域收尾 | 保留 |

原始目录：`results/ieee_tc/p2_backend_qualification/d339_20261008/`。
没有正式 SLO、N_correct、共同 warm/Resident 或 G1/G2 达标结论。
