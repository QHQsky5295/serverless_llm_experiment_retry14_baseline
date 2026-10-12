# D480：HydraServe 原生控制器的平台隔离接口

2026-10-13。属于必要平台适配，不是性能优化、GPU 正确性或完整复现结果。
不连接、不修改用户已明确为其他项目所有的 k3s；未使用管理员权限。
执行主线保持 HydraServe 实际结果 → 完整可用基线比较 → PrimeLoRA 优化。

## 来源、实际改动与边界

依据[官方 HydraServe 实现](https://github.com/LLMServe/hydraserve)的固定
commit `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`。官方部署使用 Kubernetes
和 Aliyun GPUShare；其修改后的后端、启动重叠、流水线及 consolidation
不能由普通 vLLM 代替。官方 checkout 仍完全干净。

`HydraServe_project/runtime/platform_port.py` 对 resource_manager、scheduler、
algs 三个完整文件校验 SHA，给前两个生成内存中的小幅 source overlay。
不写原 checkout、不自动安装、不运行容器，也不构造假的 Kubernetes Pod。
任何原始源码变化都拒绝生成，实际镜像内源码仍须单独核验。

仅引入 keyword-only `platform`，接口为真实物理 inventory、storage address、
worker 创建、地址查询、删除和实际退出确认。未传 platform 时保留原 Kubernetes
路径。现有源码仍可导入 Kubernetes 相关库，但选择平台接口时不应访问集群。
具体隔离容器实现和整体控制器入口尚未接入，不能把接口测试当成服务可用。

保持原样的内容包括：

- 全部 `algs.py`，Scheduler 的调度、扩缩容、队列、启动与合并逻辑；
- NetManager/NetworkEvent；资源 best-fit、<=6 GB 剩余量处理、原生资源记账；
- 原生模型/PP_RANK/PP_SIZE/DEST_PP_SIZE、加载方式和 batch 环境值；
- 原生 `/init` 请求与等待确认，而不是看见容器 IP 就宣布 engine-ready。

平台调用额外接收原 ResourceUsage 中的 gpu_id，最终须映射到实测 GPU UUID。
官方 create_deployment 只提交 GPU memory，由插件选择卡；本地实现必须尊重
原控制器已经选择的物理卡。计算出的 core_percentage 原本未施加 GPU core
限额，本适配不把它变成一个新性能策略。物理 GPU-s 单独按实际持卡区间积分，
不能用原生 GPU-memory cost 冒充本项目的生命周期 GPU-s。

## 单机四卡的关键限制

Scheduler 将每个物理节点的最大剩余单卡容量送入 `ours`。算法对 pp_size=p
要求至少 p 个可用物理节点。因此本机 inventory 应是一个节点、四张 GPU；
不能伪装成四个节点来触发论文里的跨节点模型拉取流水线。

CPU source witness 在一个物理节点、4×24 GB、14 GB 模型/原生5 GB KV项的
合成输入上执行未修改的 ours：四次分配均为 pp_size=1、node_id=0，底层
best-fit 分别使用四张卡，没有 consolidation 标志。它验证选择路径，不是
实际7B footprint/profile或性能测量，亦不是人为选定最终性能配置。

后续保留原生机制并记录自然触发情况。受控多 stage 正确性检查可单独定义，
但不能冒充本机自然多节点扩容或给出未测跨节点性能。此限制不能通过改名消除。

## 验证与失败证据

12 项新 CPU 测试，加 D479 的10项绑定测试，共22项通过。覆盖：

- 完整来源 SHA、拒绝未知版本、原 checkout 不修改；
- AST 比较确认上述非平台方法、其他类和初始化网络请求没有变化；
- 一节点四卡表示、原生卡选择和剩余量规则；
- native 环境值和 GPU identity 传递；两个**合成**节点下 PP/destination 参数保留；
- 删除未确认时不归还资源，退出核查失败时保留占用；
- 创建失败传播、不虚构地址；原生 /init 未确认时继续等待；
- 默认 Kubernetes 路径与适配路径的 native 参数一致。

初次为21通过/1 error：测试替身遗漏原生默认的 prefill_time=0，补全该字段后
重跑通过。原始错误、源码SHA、可逆的一行更正和资源关闭回执均保留；没有改
生产算法来迎合测试。测试中的地址和 worker 是惰性替身，不代表实际容器状态。

## 必须继续完成

1. 完成同一官方镜像的断点下载、完整 SHA 校验、受限导入和实际源码/依赖核验。
2. 将本接口连接到项目独占的真实容器生命周期：真实地址、独立共享内存、
   预先建立的父/子资源限制、UUID 绑定、身份核验、真实退出与 GPU 释放证明。
   复用 D477 已通过的受限容器组件，不重跑仅 NVML 的已知检查。
3. 原存储服务器会清理 /dev/shm/psm_*，必须隔离在本项目 IPC 空间；不能暴露
   host shm。原 remote_server 会加载模型并复制完整 CPU bytes，必须核验
   共同80 GiB包络和原生 packed 权重交付；不得在唯一基座上运行破坏性下载器。
4. 接入 D479 共享层 LoRA owner，验证实际原生张量、非零 adapter 数值、合并
   切换；接入请求身份、固定输出和原生 token 计时。CPU tests 不能替代这些。
5. 然后依次完成100请求、500-ID、1000请求及完整4000 W0，不提前新增
   PrimeLoRA 候选，不把下载耗时算作推理实验。

原始记录：`results/ieee_tc/hydraserve_qualification/d480_20261013/`（实验仓库）。
封存小型证据：本仓库 `paper_results/ieee_tc/hydraserve/d480/`。
官方软件下载为仍独立运行的D479；本次不重启它，不碰GPU、远端工件和旧结果。
