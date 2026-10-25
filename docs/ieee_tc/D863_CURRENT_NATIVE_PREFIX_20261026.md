# D863：当前3B短输出停顿的有界原生诊断

2026-10-26，预登记；尚未产生结果。沿用已批准主线P3，非新性能候选。

## 问题和已有证据

D862已经复用D858/D861完整4000请求，解释了首次TTFT跨在线扩容线与
GPU占用的闭环差异；D860已撤回，生产逐字节恢复D857。当前开发参照D858
联合3746/4000=93.65%，距共同95%仍差54条。D861全部93个TPOT违约均为
输出不超过64token；其TPOT-only组的同副本prefill包络重叠反而更少。
这些证据不支持直接降低token预算，也不能把剩余违约全部归于远端网络。

当前尚缺：在实际首末token窗口内，原生控制调用、engine step和其余时间
各占多少。D722复用D684原生1000请求，有312160个span、89915个step；但
早于D725图范围与D747/D751 HOST融合。D723是旧版100请求组件诊断；D843
是当前controller栈，不含所需原生逐迭代分解。三者不能替代当前测量。

本次不重复D844/D856 ctypes-stat、D851–D854 whole-owner线程、D860文件
投影、D712/D713阈值或已经完成的HOST融合方向。没有新的性能参数候选。

## 一次诊断与证伪边界

- 使用D858模型/运行配置，仅改output/NVMe/HOST三个独占路径。
- 原4000请求源、原W2 adapter map、原prompt/target/arrival均不生成新数据；
  执行既定前1000条，保留完整源身份和diagnostic_prefix_v1身份。
- max3、max-seqs8、token预算4096、graph范围、IEEE九公式、profile、
  共同SLO、真实远端已发布gzip和isal解码全部不变。
- 只启用已有host_spans_v1。不开python_frames、v2_components、HOST inventory
  子计时；不修改观察器，不添加同步、CUDA事件或读回。
- 每原生进程维持64MiB/200000span上限，记录观察开销/截断/时钟/进程身份。
  若截断，报告覆盖范围，不扩上限或重跑同键来补齐有利样本。
- 以现有分析器交叠原生首末token窗口，分控制client、单/多token分配step、
  input非client和未覆盖时间。嵌套时间不重复相加，异常和无覆盖窗口保留。
- 若控制调用占主要部分，才沿实际utility边界查重复工作；若step占主要部分，
  先查已记录分配形状及本地源码；若未覆盖部分主导，保持归因未知。三者均
  不预先决定改批处理，不把未覆盖时间直接命名为输出交付或GPU计算。

只执行一次；1000条不代表4000条的后期工作集，也不是M1/M2。观察开销留在
时延中，不与无观察器Full作性能优胜比较，不构造去掉组件后的反事实TTFT。
独立数值LoRA证明仍未知，n_correct=null、ranking_eligible=false。

## 官方实现依据

[vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的busy-loop先处理输入再执行step；step包含输出入队，异步batch还可能回收
前一batch，故主机span不是纯GPU kernel计时。它支持观察边界，不证明本机
瓶颈。[官方profiling说明](https://docs.vllm.ai/en/v0.30.0/contributing/profiling/)
明确观察会干扰性能；本次采用已有有界轻量span而非完整Torch trace，并保持
诊断/正式测量身份分开。

## 变更、资源和收尾

| 文件 | 改动 |
|---|---|
| configs/ieee_tc/20261026_d863_3b_native_prefix1000.yaml | D858三路径克隆 |
| scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh | 登记prefix、原W2与decoder入口 |
| scripts/ieee_tc/profile_d328_controller_python.sh | 登记原生span观察tag |
| tests/test_ieee_tc_d456_diagnostic_contract.py | 三项D863接口检查 |
| raw d863_20261026管理脚本与prelaunch快照 | 复用既有启动/备份/远端释放过程 |
| 生产算法、core_timing、原生后端 | 零修改 |

CPU资格/备份3/4GiB、swap0、CPU2/3/26/27；服务72/80GiB与既定外置回放和
watchdog，磁盘150/100GiB及新增峰值检查不变。不操作共享k3s，不并行维护。
147历史项和用户manifest保护；只stage明确文件，无凭据/原始大日志入库。
先检查备份再启动，结束后实际释放GPU、按本次InvocationID停止远端服务，
再分析，不开启重复诊断或画图。

完整同版本两模型W0/W1/W2、baseline/dLoRA、A1–A5、S1–S13、数值证明、
审稿图文和最终V2同步均保留。下一步由该诊断的实际覆盖和分解决定唯一候选。
