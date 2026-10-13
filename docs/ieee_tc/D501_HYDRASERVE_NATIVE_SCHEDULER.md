# D500–D501：原生 HydraServe 调度与空闲回收

2026-10-13。这是必要平台与测量资格，不是性能点，不进行基线调参。

## 真实执行

D501 从无实例开始，通过原始 `Scheduler.process_request`、排队检查、
`ours` 算法、prefetch、ResourceManager、原生 SHM fastload 和 `/init`，
完成既有 req_00000 / finance_lora（152/152 tokens）和 req_00021 /
medical_lora（89/89 tokens）。保留官方修改后 vLLM 0.4.2。
实际一台四 GPU 主机，原算法选择 pp=1；不虚构四个节点或 pipeline 触发。
请求结束后，原生 30 秒 idle 检查删除实例并恢复资源账目为 [[24,24,24,24]]。
实际 worker、provider、storage、pause 退出，GPU 无残留，不以逻辑计数代替释放。

原 ModelSet[0] 中 test 的 Llama-2-7B 注册值 (13,9000,200,1000000,1)
仅映射到既有本地 checkpoint ID；batch=8、USE_CACHE=0、SLOW_EXPR=0、
无 MAX_PP_SIZE 覆盖。这不是最终共同 SLO 或性能配置选择。

## 最小必要适配

- `controller_runtime.py` 校验八个原生模块 SHA，再加载完整控制器。
  仅把 utils 的 Kubernetes import 移到五个真正使用 SDK 的原函数内部；
  HTTP/TCP、调度、placement、原生算法不变。没有 SDK 替身；调用 Kubernetes
  路径仍需要真实 SDK。现有其他项目 k3s 不访问、不改配置、权限或资源。
- `SourceOverlay` 增加明确的 flat-module 映射，manifest/SHA 和命名冲突
  检查失败即拒绝。退出只清除本次加载模块，调用方负责先结束任务和资源。
- 原生输出容器 PID 与主机 worker PID 属于不同 PID namespace。
  `matches_worker_process` 校验容器 entry PID、实际 host birth 和 boot ID；
  实际所有权/cgroup/clock namespace 检查仍单独进行。
- 原生 Python 3.10.20 不提供 os.pidfd_open；仅安全收尾 helper 使用已有
  `/usr/bin/python3` 的 pidfd。先核对 exact scope、UID、PID/birth 和 comm，
  不退化为按裸 PID 杀进程。模型依赖环境不变。

## D500 失败保留

D500 同样从原调度器启动，第一条请求实际生成 152 tokens 后，验证器错误地
要求容器 PID=主机 PID，导致资格中止；第二条未提交。原 generation_result
保留 passed=false 和空 requests，不能追改。原始 native events 留存。
其 finally 又因原生 Python 缺 pidfd API 退出，故 result.json 不存在；
外置已资格化收尾实际释放所有资源。不能补造不存在的成功结果或退出码。
D501 只修正上述身份核验与收尾环境，不改变原生策略或性能参数。

## 资源与证据范围

共同服务 high=72/max=80 GiB、swap=2 GiB、CPU 4–23/28–47。
D500 peak=42,516,488,192 B；D501 peak=42,705,088,512 B；
两次 high/max/OOM/OOM-kill 事件和 swap 均为零。
D501 service Invocation=7d7de1bee24445e19a7158621db9bed3，inode=24043224；
aux Invocation=ec70d7d5d49045ca9287feb9236f30c3，inode=24043182。
关闭回执核验真实递归空组，不凭 active 标签判断释放。

5 个新 CPU 测试覆盖原 AST、完整 import、原算法/真实单节点资源表、
manifest/模块冲突和 PID namespace；加此前 70 个，共 75 个回归检查。
CPU 测试中的 HTTP doubles 只用于 import 测试，不冒充真实服务资格。

实际远程 LoRA、100/500-ID/1000/4000 回放、数值 LoRA 正确性、完整
pipeline/KV migration/concurrent consolidation 仍未完成。原生 binding
不是数值正确性。现有两条完整路径结果不进入主表或 G1/G2 排名。
下一步复用共同按需远程 resolver，仅在实际请求触发后物化及转换 tokenizer
元数据；保留原生选择/加载语义，不预加载整池、不复制权重或重建输入。

原始证据：results/ieee_tc/hydraserve_qualification/d500_20261013 和
d501_20261013。curated summary/归档含原失败、执行源快照和成功记录。
来源漂移检查针对各次 executed_sources，不假定当前文件就是过去执行版本。
