# D573：HydraServe 本机生命周期适配的跨副本阻塞

2026-10-16。仅 CPU 因果/回归测试，不是新的 GPU 性能结果。已有 D510/W0、
D546/W1 不覆盖、不减去估计开销，也不宣称这项问题解释全部低 SLO。

## 问题与证据

D572 从原始日志发现启动期间已有请求完成、但没有新分配；原生控制器等待
启动的行为保持不变。另一个可独立检验的问题是本地 Podman 接入层使用全局
`asyncio.Lock` 包住创建、寻址和整个关闭过程，包括 `stop --time=10`。
该锁是本机适配引入的，不属于原生放置/扩缩容算法。

两个事件屏障测试固定一个副本停在删除阶段，分别尝试另一个副本的寻址与
创建。原实现均在释放删除屏障前无法继续：13 项测试中恰好这两项失败。
测试检查事件顺序，不以任意毫秒阈值声称实际性能提升。

修改仅将容器生命周期互斥限定到每个副本。短注册临界区保证名称、地址和
所有权唯一；close 禁止新注册并等待已经登记的创建，依赖顺序仍为 worker、
provider、storage。同一副本的创建/删除/寻址保持互斥；有界 CLI 完成后才
释放锁，取消/失败记录不丢弃；实际 PID/cgroup 释放检查和 GPU union 计量不改。
没有缩短 stop 超时、移除资源限制、更改 native scheduler 或伪造 ready。

## 验证结果与局限

- 修正后 71 项 CPU 检查通过，含 21 项平台检查、原生控制器保真、资源 union、
  LoRA stage/worker/request 绑定；新增并发关闭、同副本顺序、取消与残留进程检查。
- 成功测试资源峰值 51,798,016 B；memory high/max/OOM/OOM-kill/swap 均为零。
- 首次测试加载命中了错误的 tests 包；首次扩展回归因系统 Python 缺 numpy
  及测试命名空间冲突失败。两次启动错误保留；改用既有环境/实际测试目录后
  完成验证，没有为通过测试安装依赖或修改被测算法。
- 没有调用真实 Podman、Kubernetes 或 GPU 推理；CPU 通过不代替实际生命周期
  和推理资格。下一步复用原 100-request 输入进行受影响路径的真实资格，
  合格后以新执行键补完整 W0/W1/W2。旧结果保留其原适配身份。
- 不将单物理机标成两个物理节点；逻辑分区仍共享 CPU/HOST/PCIe/NIC/存储。

原始证据：`results/ieee_tc/hydraserve_qualification/d573_20261016/`
（实际位于 `/home/qhq/serverless_llm_experiment/results/ieee_tc/`）。
备份的逐文件 SHA、测试前后源码与失败日志见本仓库
`paper_results/ieee_tc/hydraserve/d573_lifecycle_concurrency/`。

本机修改历史：929b18a 引入隔离生命周期；3eec328 修正容器整体寿命限制，
本次保留该修正。官方固定 commit
`8ae605de354ccfa2e9095514cdb4a9e9c56aa56b` 的 ResourceManager 在释放被确认后
才归还资源；本次不改变这一语义。
[官方固定源码](https://raw.githubusercontent.com/LLMServe/hydraserve/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/kubernetes/vllm/src/resource_manager.py)
与 [Python asyncio.Lock 文档](https://docs.python.org/3.10/library/asyncio-sync.html)
已联网核查。每个历史创建尾部有多少来自该锁仍未知，不由此推造反事实时延。
