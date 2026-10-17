# D641：当前3B source资格在业务前失败

2026-10-17。执行版本02f438519d2cca90aa2b710466900dac1daca2a6。
原始目录`results/ieee_tc/p2_backend_qualification/d641_20261017`。
此记录保留失败，不替换D640静态输入检查，不形成主性能点。

| 项目 | 本轮观察 |
|---|---:|
| 计划测量case | 368（复用D88索引） |
| 已开始/完成case | 0 / 0 |
| 远程获取 | 0 |
| worker_spawn至release_deferred | 180.228559757秒 |
| 服务内存峰值 | 12461797376字节 |
| 最低主机MemAvailable | 105258921984字节 |
| memory high/max/OOM/OOMkill | 均0 |
| 正式性能/Full-profile资格 | 不成立 |

## 事实、解释和未知项分开

1. 本次端口18080使用已发布只读gzip交付，启动前真实链路1000Mb/s/full。
   但尚无请求，不能由此得到传输或分层时延结论。
2. 原生日志证明3B权重加载、torch.compile和CUDA graph capture曾执行；
   权重加载21.79秒、torch.compile总计32.86秒、graph capture约9秒。
   这些跨度不是用户TTFT，也不能直接相加代替完整启动。
3. `run_d298_capacity_source.sh`仍是较早测量入口，未设
   `FAASLORA_WORKER_START_TIMEOUT_S`，因此使用spawn默认180秒。
   当前主实验`run_d293_7b_prefetch_validation_100.sh`显式设600秒。
   同时本次旧source入口未绑定D534/D535已经建立的共同native编译合同。
   它们是可核查的入口差异，不是已证实的全部失败原因。
4. 没有`ready.json`，终止路径先调用allocation.release；原生worker生命周期
   尚未确认，严格计量拒绝释放，从而最终异常为
   `native worker lifetime unqualified; physical lease retained`。这个异常
   遮住了本应显示的启动未就绪错误。保护器随后按真实GPU census清理，
   本机service已退出、GPU上下文确实释放；**不能把这补写为原owner日志
   已具有合格release**。原物理日志只有acquire、worker_spawn、release_deferred。
5. 日志最后的原生warmup完成与外层未就绪之间仍有未解释时间；本轮不把它
   武断写成GPU吞吐不足、OOM、V2 Model Runner缺陷或纯粹超时太短。

## 已完成的收尾与数据保护

业务0请求的失败完整保留。worker日志/payload/process小文件按SHA复制到本轮
证据目录，原/tmp文件未删。远端两个artifact服务及专属monitor按InvocationID
核对后关闭；monitor下载SHA一致。因没有transfer，当前clock无transfer日志
不构成丢请求，亦不伪造一次传输记录。

launcher资源域已验证递归空、关闭且路径消失；本轮starter和收集器同样关闭。
147旧投稿文件和用户manifest不变。没有从失败中取“较低GPU成本”进入排名。
无图、无新增模型/权重/trace，无其他项目k3s操作。D640准备节点已推送V2。

## 紧接主线的必要修正（尚未执行）

下一执行键只将source入口对齐到**已有**当前共同native编译合同、明确的600秒
worker启动保护和日志保留，保留D641的3B模型参数、368case、工件及真实远端。
先用既有CPU合同检查证明实际环境绑定和旧入口不变，再新键执行一次。
600秒不是重新选SLO，不改变1800秒请求保护、60秒通知或生命周期计费；全部
真实初始化持卡时间仍保留。不能修改D641原SHA/错误或在原路径重新跑。

本次产生的模型相关编译/page cache状态应记录，下一次有命中不能被全部归因
于启动保护或“系统优化”。若对齐后仍未就绪，按实际worker堆栈/阶段做有界
诊断；不再盲目加超时、不构造fallback成功、不重新打开7B资格分支。

本轮属于3B资格推进并发现入口不一致，**不是3B性能达标**。后续Full、共同warm/
Resident、3B全部baseline/dLoRA、A1–A5、S1–S13及图表/审稿文档范围仍完整ACTIVE。
