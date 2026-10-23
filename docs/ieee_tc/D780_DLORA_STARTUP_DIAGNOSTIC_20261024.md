# D780：dLoRA 3B 启动压力诊断与最小适配

## 观测，不是性能结论

真实远端完整500池准备完成，保留的原生Ray成功启动并进入DP2/TP1模型加载，
但没有native ready、合格请求结果或完整物理资源账目。因持续严重回收压力
主动结束诊断；不称OOM、不称watchdog触发，更不将其归因于dLoRA算法劣势。

服务峰值78,497,169,408 bytes，memory.high事件12,336,539；max/oom/oom_kill
均为0。停止前快照49进程，其中36个空闲Ray worker、4个原生actor。空闲
进程RSS之和16,882,376,704 bytes，匿名RSS之和9,572,519,936 bytes；共享
页可能重复计数，不能说这是可直接节省的独占物理内存。其他原生加载内存
仍然很大，因此尚不能断言取消空闲预启动就足以完成加载。

## 实际收尾与证据

先向物理分配账目绑定的父进程发送SIGTERM。原launcher在退出过程中遇到
“process membership/owner changed during census”，没有生成launch3b.json。
随后核验原服务和辅助资源域均已无进程，NVML确认无GPU计算占用，按原
InvocationID关闭这些空域；另存recovery_cleanup.json，不伪补launcher返回码
或物理release journal。远端两模型服务及本次监控也按身份关闭。

保留原生/Ray日志、失败记录、回放与分配日志、远端传输和监控证据。核对
4000文件、19,482,573,296 payload bytes与保留原池逐文件SHA一致并完成引用
核查后，仅删除本次重复objects；观测回收19,497,328,640 bytes。原池、历史
失败、145项旧投稿文件和3项用户dirty均保留。维护域受4 GiB限制，已关闭。

## 有依据的最小改动

1. Ray 2.51.2按40个逻辑CPU预启动40个Python worker。关闭私有dLoRA实例的
   enable_worker_prestart和prestart_worker_first_driver，保留CPU40、两GPU、
   8 GiB合计对象存储及原生按需actor创建；不改放置/迁移/LoRA算法和已选参数。
2. 收尾的进程列表不是原子快照。已离开所属域或owner变化的PID不应再发信号，
   也不应阻断已验证身份的cgroup收尾。新增明确记录这些PID的清理模式，
   原资格检查仍严格拒绝身份变化，pidfd和固定cgroup文件身份验证保持。
   去除原先重复枚举根cgroup.procs；不是放宽安全包络或停止其他项目进程。

依据：[Ray固定版本配置](https://github.com/ray-project/ray/blob/ray-2.51.2/src/ray/common/ray_config_def.h)、
[固定版本worker逻辑](https://github.com/ray-project/ray/blob/ray-2.51.2/src/ray/raylet/worker_pool.cc)、
[Linux cgroup进程与退出语义](https://docs.kernel.org/admin-guide/cgroup-v2.html#processes)。

现有私有Ray选项检查通过；实际native Python3.9/Ray2.51.2的CPU-only小测在
CPU40下没有空闲预启动，按需创建4个actor。GPU隐藏、测试object store80MiB，
不冒称正式8GiB模型加载验证。峰值993,918,976 bytes，无high/max/OOM/swap。
清理回归初版因重复根目录枚举失败；修正后3项通过，失败与通过日志均保留。
没有重跑不受影响的HTTP、native布局和迁移测试，没有新安装或GPU重跑。

## 下一步与范围

用新执行键回到同DP2/TP1的真实模型加载、必要GQA/LoRA及共同HTTP资格，
之后完成dLoRA限定3B W1/W2。不重新搜索基线参数；不得重用D780键或把
CPU检查写成正确性/性能实验成功。D778 native source view仍可复用，D779
环境不变，wrapper与主launcher新SHA必须记录。

Loquetier/HydraServe 3B、PrimeLoRA完整比较优化、同版本Full主表/消融、
A1–A5、S1–S13、CE/Pareto、审稿文档和最终图表全部保持待完成。共享k3s
属于其他项目，本次没有触碰。主目标active；图表仍按最新规则延期。

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d780_20261024`。
小型备份目录：`paper_results/ieee_tc/baseline_audit/d780_dlora_startup_diagnostic/`。
