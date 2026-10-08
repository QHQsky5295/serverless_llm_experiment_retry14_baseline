# D340：未使用量化组件的按需准备候选

2026-10-08。7B 开发验证；不是正式比较、不是新的算法贡献。

## 可证伪假设与边界

D339 真实导入观察发现两个 Humming 后台构建线程。当前模型为未量化 FP16，
而 Humming 是量化 GEMM 组件。假设：每次副本扩容重复启动这些可选后台准备，
会消耗 CPU/内存并与有效初始化竞争；改用组件原生按需准备可能降低扩容代价。
这个假设不等于已证明它是主瓶颈，必须由实际 activation 与完整回放检验。

只改一个底层准备策略：`ieee_quantization_prebuild_policy=on_demand_v1` 对应
`HUMMING_DISABLE_PARALLEL_BUILD=1`，在 Python 导入前生效。保留 native spawn、
独立副本、FP16/attention/LoRA、图缓存、batch/token 配置、真实远程、全部控制
参数和 IEEE 九个公式。未使用的量化组件仍可按需构建，不删除模块、不改 kernel。
不合适用于未验证的量化工作点；后续共同 vLLM baseline 可获得同一合理优化。

来源：本地 humming-kernels 0.1.12 的 `ops/utils.py`、`utils/nvrtc.py`、
`utils/jit.py`；[官方库](https://github.com/vllm-project/humming)和
[官方按需/后台构建开关](https://github.com/vllm-project/humming/blob/main/humming/utils/nvrtc.py)。
本地源码散列锁定实际行为；不把在线 main 当作本地安装版本的逐字证明。

## 验证顺序

1. CPU 回归与真实导入：无 CUDA 初始化、无 Humming reaper；native spawn 保留。
2. 复用 D336 配置，仅候选标志与新结果路径改变；复用原 4000 trace 的前 100
   请求作 smoke，**不重生成 trace**。GPU/内存/磁盘/远端检查通过后才启动。
3. 在每个实际 ready 的 frontend 检查 native unquantized 配置、原生核心进程
   方法和后台线程；原生日志继续用于 activation/模型加载/图初始化分解。
4. 全部 100 请求完成、原生 token/身份/时间分解与真实释放通过，才考虑同候选
   完整 4000 W0，与 D336 完整结果比较。短前缀不能外推 GPU-s 或证明 G1/G2。
5. 如果实际差异不支持假设，保留失败/退化并返回下一瓶颈，不叠加 fork 优化。

服务/准备 EWMA 的初始化样本仍是既有**ready 后**测量；仅这个有严格 FP16
前提的启动开关不纳入 post-ready profile 身份。完整配置、执行身份和后端
环境仍保留开关，runtime ready handshake 不删除该字段，正式 warm 阈值仍待测。
不能据此忽略 batch、LoRA、allocator 或其他推理配置变化。

## 修改与观察范围

| 文件 | 改动 |
|---|---|
| `faaslora/runtime_configuration.py` | 显式配置与导入前原生开关对应检查 |
| `scripts/run_all_experiments.py` | 初始化时检查合同，不改请求路径 |
| `scripts/dedicated_engine_worker.py` | ready 前一次性核对并记录真实后端配置 |
| `faaslora/experiment/instance_pool.py` | 有条件复用 ready 后初始化 profile |
| 现有 launcher、tests、新 D340 config | 单候选短回放，旧配置不覆盖 |

G1/G2、数值正确性、共同 warm/Resident 和旧新 Prime 完整资格仍开放；后续
顺序仍为 7B→3B→基线→主比较→消融→敏感性，不提前画图。
