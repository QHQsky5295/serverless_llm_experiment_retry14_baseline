# D502：HydraServe 原生调度与按需真实远程 LoRA

2026-10-13。继 D501 后的交付兼容资格；不是性能实验或完整池资格。
现有其他项目 Kubernetes 集群仍未访问或修改。保留官方专用 vLLM 0.4.2、
完整 Scheduler/ours/ResourceManager 和原生冷启动、SHM 加载、30 秒空闲回收。

## 必要适配及依据

原始 Request/Instance/ChatEngine 不带远程 adapter 交付。复用已在其他
系统核验的 PublishedArtifactResolver，新增 NativeArtifactViews 将其
核验后的文件提供给旧 tokenizers 格式。只转换 tokenizer JSON 元数据，
其余文件链接至本次下载对象，不二次复制权重，不再生成 LoRA 或执行负载。
官方 scheduler.py/engine.py 从 pinned commit 再次在线取得，与既有源码逐字节
相等；完整原生 Scheduler/Model/Request、placement 和 profiling AST 不变。

处理顺序是实际请求进入原调度器 → 实例分配/初始化 → 共同按需下载及元数据
物化 → 原生提交/生成。没有提前查看未来请求、整池预取、就绪排序或缓存策略
移植。下载使用共同 300 秒传输保护；原生 5 秒提交超时仍仅管后端提交，不把
新增远程 I/O 偷塞进它。请求总保护和实际用户时延不删减准备阶段。

同对象的下载复用共同 resolver 的进程间锁；元数据视图由单个控制器按对象锁
保护，原子移动到最终路径，完成后只读，再发布回执。取消等待写入结束，防止
收尾时残留写线程。已有视图/来源回执不符直接报错，不回退本地原始工件池。
原始 Instance 计数在准备失败/取消时由已核验的异常路径释放。

CPU 初版三项测试发现 readonly 暂存目录跨父目录 rename 的权限要求；在受控
暂存阶段保留 owner write、最终发布后封只读修复。失败说明留存，未启动 GPU。
修正后 80 个 CPU 回归检查通过；没有基线性能参数搜索。

## 实际证据

复用 D498 已 canonical 化的两条既有源 trace 记录，非新工作负载：

| 请求/adapter | 原生输出/目标 | 压缩线上字节 | 交付性质 |
|---|---:|---:|---|
| req_00000 / finance_lora | 152/152 | 14,718,801 | request-driven cold miss |
| req_00021 / medical_lora | 89/89 | 14,736,516 | request-driven cold miss |

远端 transfer_id 与本地回执、内容 SHA、已发布归档 SHA 一一核对。
两次 remote pack_performed=false；temporary_created=false；没有请求中打包。
本次网络读取真实记录 1000 Mb/s/full，不据此声称历史 100 Mb/s 的原因已解决。
远端共用 D78/D80 只读发布缓存，没有重新发布、下载新权重或复制完整工件池。
静态 500-ID 描述只作为配置索引，不触发下载；本轮实际对象只有上述两个。
这里“复制权重=0”仅指元数据视图不二次复制；实际远程下载字节仍计入交付。

本地必要读取、接收、解包、内容核验、元数据转换与原生加载均属于实际请求
路径，不从 TTFT/E2E 中事后减去。两机 monotonic 时间域分开，不直接相减。
原生输出 binding/计数与 D501 的输出序列核对只是集成证据，不是 FP 数值证明。

原生空闲回收实际删除 worker，rgpus 恢复 [[24,24,24,24]]；provider、storage、
pause 均退出，GPU 和服务资源组清空。服务 peak=42,970,333,184 B；
high/max/OOM/OOM-kill 与 swap 均为零。服务 Invocation
9d1230e11d7d482e9b882d6e8dc0a2ae，inode24053924；辅助9 Invocation
728b0375b7c34d2d87d98d1e14c8693c，inode24053882，实际空组关闭。
远端精确核对本次 Invocation 后才停止并收集日志，发生在本机释放之后；
两服务和监控已 inactive，端口18080/18081无监听，日志 SHA 与远端一致。

## 下一主线及限制

仍需原生 HTTP 前端/共同 open-loop 回放、物理 GPU 生命周期计量对接、数值
LoRA 正确性，以及 100 → 500-ID 功能覆盖 → 1000 → 4000 的实际结果。
既有两条测试不能冒充 100-request smoke 或完整性能结果，不能据此填主表。
完整 pipeline/KV migration/concurrent consolidation 仍未资格化；本次为原算法
在实际单节点上自然选择 pp=1，不强行制造机制触发。
之后完整可用基线比较，再优化 PrimeLoRA；不提前新增 PrimeLoRA 候选。

原始记录：results/ieee_tc/hydraserve_qualification/d502_20261013。
curated archive 只收脚本、源快照、回执和日志，不含下载权重、token、私钥。
D500失败、D501成功及所有旧投稿结果保持原样。
