# D338：启动导入诊断——不接受“删掉钩子即可节省 10 秒”的归因

日期：2026-10-08。阶段：P2/P3，仍为 7B 开发优化。没有新增 GPU 回放，
没有修改服务算法、后端配置、论文、权重、trace、指标协议或绘图。

## 问题与历史依据

D337 从 D336 的 11 个实际 native worker 日志得到平均 activation→ready
47.371 秒；model-load 平均 9.379 秒，profile/KV/warmup 平均 11.501 秒。
这些嵌套、部分舍入的 span 不能简单相加或从总时间中扣除。
D335→D336 新增的 30 个暂定联合时延违约中，29 个落在两次运行启动窗口
并集的共同请求集合中；这是分组描述，不是因果解释比例。

本次唯一问题：重复 Python/frontend/core 导入是否存在可以直接移除的启动成本？
代码历史发现，`sitecustomize.py` 的 GPU worker shutdown 补丁来自
`e717f80011ac603810ebcf0b3ed040f9479b1dfe`。它在每个继承项目 PYTHONPATH
的 Python 进程入口导入 torch 和 `vllm.v1.worker.gpu_worker`。
D69 曾记录同一入口影响外置 publisher，不能把它当作首次发现的机制。
当前实际 frontend 还会经 `coordination.__init__ → coordinator → serving`
导入 vLLM；顶层 `faaslora.__init__` 已经是 lazy exports，不应重复实现。

## 诊断合同与实测

使用实际 native Python 3.12.12 / torch 2.13.0+cu130 环境，在独立
MemoryHigh=3 GiB、MemoryMax=4 GiB、swap=0 资源域执行两个顺序进程。
实际 affinity 均为 2,3,26,27；CUDA_VISIBLE_DEVICES 为空，两进程结束时
torch.cuda.is_initialized() 均为 false，前后 GPU compute census 均为空。
不构造引擎、不读取权重、不传输 adapter。使用 Python `-X importtime`。

| 阶段，秒 | 原项目启动钩子 | 中性入口后导入相同 frontend |
|---|---:|---:|
| 进程创建到测量脚本入口 | 10.163524 | 0.023166 |
| 入口后的 runner 导入 | 0.661289 | 12.791289 |
| 随后的显式 frontend import | 0.000038 | 0.000029 |
| 整进程 wall（含退出） | 13.004194 | 14.404769 |

整进程 wall 的单次差为 +1.400575 秒（+10.77%），**不能**据此宣称中性入口
显著更慢，也不能把 10.16 秒当作可消除的 startup/TTFT。这里只说明，
绕过早期导入没有在此诊断中缩短完成相同 frontend 导入的总时间。
第二种入口省略了启动兼容补丁，只用于定位，绝不是已通过资格的服务版本。

每个路径仅一个观测、固定顺序、使用较小 CPU 集合，并包含 importtime
记录开销；不等同于 40 逻辑 CPU 服务中的并发扩容。累计 import 时间嵌套，
不能相加。退出耗时也不能当作 ready 的关键路径。

资源峰值 1,686,216,704 B；high/max/OOM/OOM-kill 均 0，swap 0。
最初辅助启动在读取未暴露的 `cpuset.cpus.effective` 时退出，尚未执行
native 导入；保存为 measurement/launcher error，修正读取方式后记录
实际继承 affinity，不伪造 cpuset 委派。失败 scope 已核验空并停止。

## 原始实现与文献核对

- vLLM 0.30 的 [CoreEngineProcManager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/utils.py)
  通过独立进程启动 EngineCore；不能用 frontend 的 Python 导入时间直接
  替代 core 初始化时间。
- [core_client](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)
  显式处理进程启动与后台 warmup 顺序，避免线程锁与 fork 交叠。
- 本机 0.30 `EngineCore.shutdown()` 已调用 `cleanup_dist_env_and_memory()`；
  旧项目补丁不能未经验证就移植或删除，也不能为省启动时间 fork 已初始化 CUDA
  的进程。对应[官方 core 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)。
- Python [importtime 文档](https://docs.python.org/3/using/cmdline.html#cmdoption-X)
  区分 self 与 cumulative 时间；本次只在单个被测进程中解释这些记录。

`serving-llms-vllm` 技能用于定位后端边界；其中通用示例阈值不替代冻结 V1。
`analyze-results` 用于将观察、解释与下一步分开；最新用户指令要求达标前不画图。

## 裁决与下一步

1. **不采用单纯移除 sitecustomize 的优化。**现有诊断没有支持其完整启动收益，
   更不能扣减历史延迟或资源时间。没有叠加新的兼容补丁。
2. 启动仍是值得优化的端到端问题；下一步必须针对实际 worker 的
   CPU 初始化→引擎构建→core 启动→ready 边界，而不是继续重复导入微测、
   调整容量阈值、重复 512-token budget 或关闭 CUDA Graph。
3. 新 GPU 诊断前先恢复磁盘启动余量：当前约 149.6 GiB，低于冻结 150 GiB。
   本次仅只读检查缓存，未删除文件。uv 大文件有多重硬链接，目录表观大小
   不等于能释放的空间；不能按目录名删除。保留原始结果、工件、在用环境
   及当前编译缓存。
4. 7B G1/G2、N_correct、正式共同 warm/Resident 与匹配旧新资格仍未关闭；
   下一阶段 3B、外部 baselines、正式比较和消融继续待办。不得把本 CPU
   诊断记作完成一个论文性能实验。

数据：`paper_results/ieee_tc/p2_backend/d338_startup_import/`。
原始：`results/ieee_tc/p2_backend_qualification/d338_20261008/`。
执行代码 cfe420d13678c67c4d2f123879634292187a8c32；源文件 SHA、完整导入日志、
失败与资源清理回执随证据包保留。既有一次性远端只读交付缓存沿用，未重建。
