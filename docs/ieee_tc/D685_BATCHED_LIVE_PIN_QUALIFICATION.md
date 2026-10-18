# D685：一次新鲜 HOST 观测内的原生批量 pin 查询

2026-10-18。3B 优化的一个有界组件候选，非性能主结果，尚未接入生产。
完整结果仍 D682：联合 SLO 3694/4000、92.350%，U=15896.253912 GPU-s，
独立数值正确性未知。九公式、共同阈值、工件/负载、基线与远端不变。

## 依据与与旧候选的区别

- D684 已完成当前版本前1000诊断；8686次新鲜清点平均74.234ms，CPU/wall
  99.24%。32适配器的完整图含7168个实际分配，不能把它当小型状态字典。
  与短decode违约的交叠是关联，不是可以从TTFT/TPOT减去的因果时间。
- D329 的对象替换、resize、alias、pinning、隐藏字段、移除反例继续成立；
  D144 每view一次、D202无view表、D671独立图复制及D675/D680事务优化不重复。
- D461定位原生pin查询；D462每view Python CUDA binding 的中位完整清点
  反而较慢，已拒绝。本次不再测这一接口替换，也不根据其异常长点选择均值。
- [PyTorch Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
  查询storage基址；[CUDAHooks.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/cuda/detail/CUDAHooks.cpp)
  每view经通用上下文发现再查CUDA属性。
- [vLLM原生manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  仍为对象权威来源；不以逻辑ID推断物理状态。
- [CUDA Runtime](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__UNIFIED.html)
  支持逐指针实时属性，不提供本候选一定更快的保证。
  [ctypes文档](https://docs.python.org/3/library/ctypes.html#ctypes.PyDLL)
  说明PyDLL调用保留GIL；本限定serial-owner微测不引入新的Python并发窗口。

## 单一可证伪假设与预先固定规则

在同一个已初始化、单设备、串行owner的完整观测内，将所有view的CUDA
属性查询放进一个native循环，可以减少逐view跨语言封装/上下文发现，
而非减少实时检查。仍逐view查询，不按唯一指针去重；没有跨调用缓存。
只为诊断增加一个小型C++共享库，使用已存在wheel的CUDA头文件和runtime，
不下载依赖、不编译模型kernels、不增加模型、LoRA池或trace。

复用D462的backend-pin-check、tiny张量、实际注册/注销、四种图语义及20组
交替调用。新增显式--pin-bulk-library才进入此候选；默认D462路径不变。
所有逐张量元数据/容量/隐藏字段检查保留。AST克隆只将pin读延后到同一
观测末、发布图之前；局部占位值不得返回，最终检查每个alias的实际结果。
query错误、少结果、非boolean结果均拒绝，不把失败转换成unpinned。

CPU回归复用原footprint/routing/freshness反例，原生资格复用9个实际CUDA
状态及4种完整图比较。计时包括全部遍历、结果数组构造、入口guard与返回图，
不只计C函数。不生成CI、不反复测试直到均值有利，不当作系统性能运行。
若逐字段不等价或多数配对/中位完整开销没有改善，归档拒绝，不启动回放。
通过也只允许进入后续生产设计/资格，不能宣布G1/G2达标。

## 安全、来源和范围

先保存D684终态与历史复用证据，再用现有3/4GiB辅助域CPU测试/单进程构建；
原生微测使用现有受限服务及独立watchdog。无backbone、remote操作或请求。
推理机150/100GiB、实际CUDA释放、147项旧投稿及用户修改保护继续有效。
原始目录results/ieee_tc/p2_backend_qualification/d685_20261018/，小型汇总
paper_results/ieee_tc/p2_backend/d685_batched_live_pin/，不覆盖D462/D684。

本次变更清单：资格入口ieee_tc_preflight.py增加opt-in；新增诊断
host_pin_batch_witness.py/native_pin_batch.cpp及CPU回归；无faaslora生产修改。
全部3B/同版本7B场景、基线、dLoRA、A1–A5/S1–S13、CE和审稿交付仍开放。

## 已完成结果：语义通过，未采用生产候选

57项CPU检查通过（包含复用的原始检查），仅编译15,664字节诊断库。
一次实际原生资格完成：9种CUDA状态、4种完整图逐字段一致，实际
cudaHostRegister/Unregister前后均与原生查询一致。runtime13040、driver13000，
只映射原wheel的libcudart.so.13；无backbone、远端操作或工件变更。

| 完整图局部调用，20个预先固定配对 | 原生 | 批量候选 |
|---|---:|---:|
| 平均wall，ms | 9.474980 | 15.451956 |
| 中位wall，ms | 9.250762 | 8.858872 |
| 最大wall，ms | 12.792508 | 137.280433 |
| 配对中较快次数 | 1 | 19 |

配对比值中位数0.951720，表示多数局部调用约少4.83%；但候选首次计时
出现137.280433ms停顿，导致总计/均值更差。该停顿原因未测明，不删除、不
称已证明外部干扰，也不据其一个点声称所有调用都慢。没有CI，不从这些
微小图调用外推完整3B已改善、稳定加速或达到SLO。

初次离线分析错误地假定“平均更慢意味着多数配对更慢”，断言失败。
原脚本/失败资源闭合保留；analyze2仅修正分析假定，未重跑原生测试或改数据。
对用户的初步判断已明确更正。当前决定为**不接入生产，也不启动前缀/完整
负载**：中位小幅收益与未知长停顿并存，尚不能判断对短decode尾部有利。

下一最多一次有界原因核查：当前候选每view新增一个含Tensor的Python tuple，
这与原始路径的分配行为不同；需直接观测分配/GC时间，不能先认定137ms来自
GC后打补丁。复用同tiny fixture和现有资格入口，仅记录原因，不扫参数、
不为挑一个好均值重复性能运行。无明确新证据则结束此局部方向，返回当前
完整调用路径的语义保持优化；不得回到D462或改变SLO。

服务已正常退出、GPU上下文释放，辅助域全部递归空关闭。资格/启动/两次
分析均无high/max/OOM/swap事件；147项历史保护文件和用户manifest未变。
此次生产源码完全不变，最新完整成绩仍D682，不冒称新的系统实验完成。
