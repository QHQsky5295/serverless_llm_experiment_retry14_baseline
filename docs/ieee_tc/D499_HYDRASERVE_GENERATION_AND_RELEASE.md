# D498–D499：原生生成与加载缓冲区释放

2026-10-13。这是必要适配资格，不是 HydraServe 的完整性能结果。
执行顺序仍为 HydraServe 实际结果 → 全部可用基线比较 → PrimeLoRA 优化。
不接触其他项目的 k3s，保留官方修改版 vLLM 0.4.2。

## 已完成的实际实验

| 运行 | 唯一新增条件 | 实际观察 | 尚不能推出 |
|---|---|---|---|
| D498 | D498 CPU 已验证的无损 tokenizer JSON 格式视图 | 原生 Request → Instance → ChatEngine → 原始 GPU app；finance 152/152、medical 89/89 tokens；身份、计数、时钟检查通过 | 完整 Scheduler、远程获取、数值正确性、性能排名 |
| D499 | 原有 SHM 控制连接改为显式拥有的非阻塞 socket | 加载后、实例仍存活且尚未生成时，原生查询返回 1（已知对象已清空），并已出现原生 free_buffer；随后同两请求完成 152/152、89/89 | 多副本、pipeline/KV 迁移、并发 consolidation 的正确性 |

输入复用原始 seed42 trace 中两个既有非零 LoRA 的首条请求，不新造请求或权重。
共同 canonical prompt IDs，greedy/ignore-EOS/no-stop，原生输出 token IDs；
期望长度不代替实际计数。单进程实际身份、启动时刻与时钟命名空间匹配。
这是两个请求的限定资格，不称 100-request smoke 或 4,000-request 回放。
原生绑定确认不是与 PEFT 参考的数值等价证明。

D498 曾保存的 CPU-only curated summary/archive 保持原样；其中“GPU 尚未运行”
是当时的时间边界。本次 GPU 证据另存 D499 包，不追改历史。
D497 首次生成失败及所有原始日志继续保留。

## 原因、依据与最小修复

原生 app 在模块导入阶段以 `asyncio.run(start_load_model_shm(...))` 打开
StreamWriter；该次调用随后关闭事件循环。实际 `/init` 在另一个循环中调用
`end_load_model_shm`，D498 记录 `Event loop is closed`，存储释放仅在删除
worker 时观察到。这是加载控制连接的所有权问题，不能解释为模型推理慢。

该因果判断与 [Python 3.10 asyncio.run 文档](https://docs.python.org/3.10/library/asyncio-task.html#asyncio.run)
的循环生命周期一致；[非阻塞 socket 接口](https://docs.python.org/3.10/library/asyncio-eventloop.html#working-with-socket-objects-directly)
用于当前循环中的待完成 I/O，而 socket 自身由 `NativeShmLease` 持有。
对应 [官方 app](https://github.com/LLMServe/HydraServe/blob/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/image/app.py)
SHA 为 `cab8bc3195159a51da0c4e8ca332a38b049f65891e49cdcd2534fe984235717f`。

修复保持原始 header、EOF 释放协议、C++ loader 参数和调用线程、导入/加载
重叠、cache 策略、调度策略及原有 `/init`/consolidation 调用点。只替换两个
连接函数及 owner 声明，启用参数为 `shm_control=True`；默认旧路径不变。
取消、连接/描述符/loader 错误均关闭本次连接并重新抛出，不把失败伪装成 ready。
重复 begin 拒绝且不关闭原 lease；原生不带长度的最多 50-byte 名称格式保留，
不声称修复了其通用网络分帧问题。没有延时注入、定期清缓存或扩大内存兜底。

D499 原生 free_buffer 区间为 `(0, 13476855675)`，即本次既有 checkpoint
的 packed payload，不是新设置的容量。查询与日志都在 worker 删除前取得；
并核对 worker PID/start_ticks 仍存活。这里证明的是 SHM 池区间可重新分配，
不是宣称已将预分配 tmpfs 物理页归还主机或降低了整个服务内存峰值。

## 安全、回归与保存

- 7 个新增 CPU 测试使用真实 loopback TCP，覆盖跨循环 EOF、截断、取消、
  重复 begin、再次申请、loader 失败和非目标 AST 不变；完整回归在本包保存。
- D498 服务峰值 42,909,892,608 B；D499 43,116,400,640 B；两轮均无
  high/max/OOM/OOM-kill/swap。共同 72/80 GiB、swap 2 GiB、CPU 边界未变。
- 两轮 provider/storage/worker/pause 已退出，GPU 实际空闲，服务域已消失；
  辅助域按 InvocationID/inode 与实际递归空成员核对后关闭。
- D499 初次准备因引号 anchor 不匹配，在生成 probe 前失败；保留原 prepare，
  后续 finish_prepare 仅完成缺失准备文件，不覆盖已经执行的运行证据。
- 只复用 D498 只读 tokenizer 元数据视图与原权重链接。没有复制模型/工件池、
  新建负载或改变真实远端。旧投稿保护清单与用户修改须在备份门槛再次核验。

原始证据位于原项目 `results/ieee_tc/hydraserve_qualification/d498_20261013`
及 `d499_20261013`；本次 append-only 小型证据包位于
`paper_results/ieee_tc/hydraserve/d499/`。执行源快照与当前开发源分开核验。

## 下一项主线（不是再做环境/import 检查）

1. 将已经验证的原生请求合同接到完整 Scheduler 的实际队列/冷启动/释放路径，
   复用 isolated platform。保留 `ours` 算法、一台物理主机四张卡和原始模型
   配置语义，不能把四卡伪装成四节点来人为触发 pipeline。
2. 对接既有请求驱动的真实远端 LoRA materialization 与只读 tokenizer 视图，
   保留 cold per-run cache、内容 SHA、原生绑定和计量；不进行请求中打包。
3. 完成 100 → 500-ID → 1,000 → 4,000 的适用资格/回放，并补必要的数值、
   pipeline/KV 迁移/consolidation 证据；未触发/不适用部分明确标注。
4. 取得 HydraServe 的实际结果后再合并已有基线、定位 PrimeLoRA 的差距。
   7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13 和最终交付均未取消。
