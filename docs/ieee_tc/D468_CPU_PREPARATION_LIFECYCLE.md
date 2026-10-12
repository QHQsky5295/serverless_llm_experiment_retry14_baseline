# D468：CPU 提前准备的所有权与计量边界

2026-10-12。7B PrimeLoRA 优化主线，父版本 `767e4d4`。本次完成的是
**提前准备机制的生命周期基础及 CPU 回归验证**，不是新的 GPU 性能实验。
实际主 runner 尚未启用准备池；下一步仍须接入原扩容工厂并进行 native 资格。

## 假设、历史与不变条件

D465 完整 4000 W0 联合 SLO 为 3790/4000（94.75%），物理 GPU 时间
11984.148429 GPU-s，没有达到共同 SLO，也没有降低相对 D458 的资源占用。
D466 的完整时间线显示真实容量积压，不能归因为一个未经证实的漏唤醒。
D467 已验证固定真实 UUID 下，现有前端可到达不创建 CUDA context 的
pre-initialize 边界；它没有验证提前准备的服务收益。

本候选假设：在共同部署通知后、既有服务 CPU/HOST 预算内提前导入前端代码，
可与现有副本推理重叠，减少随后扩容的串行等待。假设可能不成立：准备的 CPU
竞争、额外 HOST 常驻和 core 内仍需完成的初始化，都可能抵消局部收益。
只有下一次新执行键完整回放才能决定是否接受；不从旧延迟中减去导入耗时。

- 保留 IEEE 九个公式、调度/驻留/准入语义、SLO、并发容量和现有配置。
- D463 owner projection 仍作为父版本的一部分，**未接受其性能收益**；不捆绑回滚。
- 不重复 D338 删除导入、D339 fork 或 D340/D341 关闭后台预构建的旧尝试。
- 仍用固定 UUID、fresh interpreter、原生 core `spawn`。不迟绑定可见设备、
  不 fork 已有线程的进程、不把 GPU sleep 计成资源释放。
- 不改已封存 baseline，不重新生成工件/负载、不访问远端、不绘图。

## 官方依据与适用边界

[vLLM 0.30 CoreEngineProcManager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/utils.py)
将前端与 native core 进程管理分开。利用这个分离并不意味着 core 内的初始化
成本被消除。[CUDA platform 实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/platforms/cuda.py)
包含按逻辑设备缓存的能力信息，因此本设计不在导入后更换 UUID。
[多进程设计说明](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/design/multiprocessing.md)
说明线程、CUDA 与 fork 的限制；历史设计说明不能代替当前安装版本的验证。

本轮网页工具成功读取 core-manager 和多进程说明，随后搜索服务返回 429；
证据包另保存公开官方原始文件的直接读取、URL、时间和 SHA。不绕过登录或
权限限制。`serving-llms-vllm` 用于核查上述边界，不采用技能中的通用性能阈值。

## 此次实现

1. `faaslora/cpu_frontend.py`：基于原合法设备集合的有界准备池、
   准备进程身份及单次 activation capability。没有 native 导入、GPU allocation、
   ready 容量或路由接口。领取和归还严格对应一个进程；实际物理释放和进程退出
   前不得生成该设备的新准备进程。池关闭后不补充准备进程。
2. 原 `dedicated_engine_worker.py` 增加 opt-in 的继承 socket 通道：完成前端
   导入、CPU context/设备核查后单独写 `cpu_prepared.json`；这不是原来的
   `ready.json`。只有验证本服务实际 acquire/bind journal 后才调用原 initialize。
   通道直接 EOF 表示取消，无需调用可能触发 CUDA 清理的 descriptor.shutdown。
   拒绝缺失通道、错误 UUID/config/出生身份、过大/重复消息和已归还的 lease。
3. 原 `PhysicalGPUAllocation.bind_process` 支持 `prepared_worker_bind` 事件，
   保存真实早期出生身份，而不是把绑定时刻伪称为新进程出生。原 journal reducer
   校验新事件；**GPU 积分仍只使用实际 acquire/release**。旧 worker_spawn 保持原义。
4. 池关闭等待已拥有的取消任务完成；已领取、可能持有 GPU 的进程仍交原物理
   runtime 收尾，不由 CPU 池猜测释放。失败保留所有权，不自动启动替代进程重试。

本轮没有修改 `run_all_experiments.py` 的实际工厂或配置，所以当前主实验路径
不会启用这些新增入口。默认 worker 路径仍直接使用原 initialize/服务/退出流程。

## 验证与证据边界

首轮 CPU 回归 554 项通过，0 失败/错误/跳过。之后补充 4 项实际 worker 控制流
测试并收紧 native/fixed-output 门槛；第二轮 **558 项通过，0 失败/错误/跳过**。
首轮源码快照、测试日志和回执保留，后续结果不覆盖首轮。

新测试使用真实本地 socket-pair/临时 journal 和模拟进程、模拟 GPU census：

- 部署通知、出生、acquire/bind 的时序及 UUID/配置/owner 一致性；
- EOF、取消、破损/过大/重复消息、已释放/其他服务 journal；
- 唯一领取、实际归还前禁止补充、部分启动失败、取消关闭和失败所有权保留；
- 原 worker 的 gate 拒绝/取消不调用 initialize 或 CUDA shutdown；
- acquire=1、CPU 出生=0.2、bind=1.5、模拟 native ready=3、release=8 的
  fixture 仍积分为 **7 GPU-s**，不能改成 5 GPU-s。

它们不是 CUDA 实测，也不是数值正确性或性能结果；不能把模拟生命周期字段当
实际 worker 证据。各轮测试集合重叠，不累计为独立实验。首轮/第二轮辅助
内存峰值为 2,299,133,952 / 1,943,801,856 B，high/max/OOM/swap 均为 0。
两个 scope 按 captured InvocationID、inode 和实际递归空进程列表关闭。
源 SHA 和 147 项历史保护核查在回执中保存。

发布前独立 system-Python smoke 暴露初始模块位置的包初始化依赖：
`faaslora.experiment.__init__` 会导入服务栈，在纯系统环境中缺少 NumPy。
首次发布已在 stage/commit/push 前停止，13 项中的 4 项导入错误和 9 项通过
全部保留。不安装全局依赖或跳过测试；将新增工具移至已有轻量包根的独立模块，
保持生命周期算法不变，并增加实际 `python -I` 导入不引入服务栈的测试。
调整后最终 **559 项通过，0 失败/错误/跳过**；辅助内存峰值 2,213,548,032 B，
high/max/OOM/swap 均为 0，按原身份和实际空进程列表关闭。
最终验证与来源包另存 `d468_cpu_lifecycle_v2/`，不覆盖原 curation。

## 接入实际扩容前仍必须完成

1. 原 `SubprocessInferenceEngineProxy` 的准备生产者：实际服务资源域、固定 UUID
   在 fresh interpreter 之前生效，socket-pair 仅传入指定 FD，出生身份核查。
   启动中失败也必须保留进程归属，并核查 CPU 导入可能产生的辅助子进程退出。
2. 原 ScenarioRunner 的共同部署通知后启动准备池、正常扩容领取、真实释放后
   补充、最终收尾先禁止补充。必须复用现有唯一工厂，不能新增第二套模型启动器。
3. 每个实际扩容仍先取得物理排他 allocation，再允许 native 初始化；若 CPU
   尚未准备完，已持卡的等待照常计量。不能先等 CPU-ready 再补写 acquire。
4. 原生资格核验两个方向：等待中的 CPU-prepared 没有 GPU context/ready 容量；
   被激活后实际 UUID、model/adapter、token、spawn、完整计费和清理均正确。
5. 完成最小 native 资格后，仅一次新版本原 4000 W0，按联合 SLO 和生命周期
   GPU-s 验收；附准备数量/用到比例/CPU-HOST开销。保留非获胜结果。

7B 尚未闭口，W1/W2/G2、HydraServe 隔离原生资格、3B、A1–A5、S1–S13、
成本补充与审稿文档/后期图表全部保留。未修改不属于本项目的 k3s 集群。

原始目录：`results/ieee_tc/p2_backend_qualification/d468_20261012/`。
curated：`paper_results/ieee_tc/p2_backend/d468_cpu_lifecycle_v2/`；
原初始 curation `d468_cpu_lifecycle/` 保留为修改前证据，不作为最终源码资格。
