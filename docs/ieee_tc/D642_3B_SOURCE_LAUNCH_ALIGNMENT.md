# D642：3B source入口与当前native启动合同对齐

2026-10-17。D641在0请求时未就绪，保留失败及真实清理记录。本次只新增
`run_d298_capacity_source.sh d642`，不改变历史分支、模型配置、368个既有case、
权重、真实远端交付和九公式，不把它当作Full性能优化。

新增入口采用已有D534共同编译合同，SHA
`328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c`，
以及主实验已经使用的600秒worker启动保护、worker日志保留。1800秒请求保护、
60秒准备和SLO没有修改。D640 source spec原样复用；其临时workspace parent
仍为D641目录，但每次唯一临时子目录，不覆盖D641结果或原始证据。

核查D641日志、源码历史02f4385/877cfc3及当前主入口：旧source未绑定共同
编译身份且默认等待180秒。可证伪假设为：对齐当前有效启动条件后，原参数
可以完成原生初始化并开始source测量；若仍失败，保留实际阶段和堆栈，不再
盲目延长等待。此次不是证明“旧失败全部由180秒造成”。D641已经产生编译/
page cache状态，后续启动变化不能全部归因于此修改。

官方依据：[FlashInfer构建和锁](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/core.py)、
[Ninja命令身份](https://ninja-build.org/manual.html)：构建输出依赖命令和参数。
本机0.6.18.post1，在线tag只支持机制说明，不声称文件字节一致。复用D535
双方共同软件产物，不进行新安装、不提前模型推理、不扣除真实初始化持卡。

CPU资格：8项共享合同测试通过；实际run_all_experiments导入入口与子进程
绑定回执相同，CUDA未初始化；删除新增分支/绑定后脚本与原HEAD逐字相同。
config/spec SHA与D640相同，147投稿保护文件未变。完整回执与后续原生
结果按不同身份保存；CPU通过不代表原生case或主性能达标。

后续保持主线：完成3B原生source资格→必要初始化和warm/Resident→3B Full及
全部baseline/dLoRA→既定消融/机制/敏感性。7B有界观察已保留，不重开HF循环。
