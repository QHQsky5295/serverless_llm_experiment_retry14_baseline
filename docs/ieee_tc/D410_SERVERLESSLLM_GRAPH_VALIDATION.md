# D410：ServerlessLLM 原始 1,000 请求验证

状态：2026-10-11 02:40（Asia/Shanghai）启动，运行中；没有最终性能结论。
这是原有开发资格流程的一步，不是新的完整主实验，也不是重复取最佳值。

## 问题和复用边界

D408 的 eager 模式在原始 1,000 请求上完成全部原生请求合同，但共同联合
SLO 为 67.4%。D409 的 100 请求原生 Graph 路径检查观察到更低的解码时间，
同时启动仍产生很长的首 token 等待。因此本轮只验证同一公开执行模式在
原始 1,000 请求（含后续轮换）上能否保持该改善；不把短前缀当成完整排名。

复用 D406 的 500 个逻辑 adapter 功能检查、D409 的受限独立参考证据，
以及 D78/D80 的已发布只读压缩交付缓存。用户再次批准一次性交付缓存；
已有缓存不重复生成，不增加解压工件池，不改变权重或到达/生成工作量。
请求触发的读取、传输、竞争、本地物化仍计入时延，请求路径不打包。

沿用原生 vLLM 0.10.2、FP16、store loader、round-robin、已披露的轮询和
请求身份队列修复；min=1/max=4/target=4/keep_alive=10，max-seqs=4、
max-LoRAs=4、CPU-LoRAs=4、token budget=1024、GPU memory fraction=0.72。
显式 enforce_eager=false 与 D409 相同，没有新策略或额外参数扫描。

本轮 LoRA 下载缓存从冷状态开始；原生编译**磁盘**缓存保留，75,240,459
字节、290 文件，内容聚合 SHA 为
`f8c62659046c116bea0ce79501c52c8824979ca8a676bb5c04cb74671cf7b9a8`。
这不是 HOST/KV/GPU 状态或预加载，不称与 D409 编译缓存初态完全相同。
是否命中由原生日志判定；实际编译、图捕获、准备及持卡开销一律保留。

依据是固定版本的[编译缓存实现](https://docs.vllm.ai/en/v0.10.2/api/vllm/compilation/backends.html)
及[官方编译说明](https://vllm.ai/blog/2025-08-20-torch-compile)，不假设跨配置缓存必然有效。

## 冻结口径与运行身份

- 完整计划 SHA：`f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5`。
- V1 和 single-run V2 共同生效，共用 D394 的 7B SLO 阈值；不改 60 秒准备
  窗口、1,800 秒资格保护或 95% 联合要求，不重复运行计算 CI。
- source trace seed=42、development identity=41、非正式 `trace_prefix_v1`；
  没有 adapter remapping。完整 4,000 请求结果仍待执行。
- main 执行版本：`d6fbaaf245f236e6f05babb2f1b7f7919e8e02f7`。
- baseline 执行版本：`336576bc8a3e7aca34febad52372911289a9ca87`。
- 原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d410_20261011`。
- 原始目录保留 `PROTOCOL.md`、配置、源视图、初态/预检、远端启动与健康
  回执，以及运行中的 `launch7b_repaired.launch/`；详细身份见执行状态。

资源边界沿用服务 72/80 GiB、swap 2 GiB，外置回放/监测 4 GiB、swap 0，
既定 CPU 分离。一次一个重型实验；实际四卡 store 持有至释放全部计量。
内存保护和磁盘保护未放宽，不在业务回放中操作远端或进行哈希/清理压力测试。

## 收尾与下一步

等待当前句柄结束，不重新启动。确认实际 GPU 释放后恢复精确 overlay 原像、
按 InvocationID 停止本轮远端服务，保存原生 worker、resolver、远端传输及
资源日志。离线核验全部 1,000 请求的原生输出、adapter/内容绑定、时间和资源；
原始未知数值正确性字段不覆写。保留所有失败、SLO 违约及成本。

校验后封存并备份，返回 4,000 请求 W0 主线；不在此扩展无关的局部调参。
7B PrimeLoRA D371 保持不变，之后继续其他基线，再推进 3B、消融及敏感性。

## D409 已关闭缓存的空间回收

此前 D409 原始数据已核验封存。仅删除已核验、可重建的
`/tmp/tc-d409-7br/remote_artifacts/objects`，回收 725,557,248 字节；
原始池、远端不可变缓存、原始记录、收据和日志均保留。删除前核对内容与
原文件、不同 inode、运行关闭和可读进程引用；三个不可读同 UID 进程条目
明确记录为限制，不宣称全系统可见。两个 cleanup 回执保存在 D409 原始目录，
历史证据清单不改写。不得为了重复已完成校验而重新下载整批工件。
