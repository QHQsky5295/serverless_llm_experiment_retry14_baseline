# D481：真实容器共享内存、主机通信与原生副本身份

2026-10-13 00:40 CST 前后。必要平台适配；不是性能调参或 HydraServe 复现完成。
执行顺序保持 HydraServe 实际结果 → 全部可用基线比较 → PrimeLoRA 优化。

## 为什么需要这一步

原生 local_server 会清理 /dev/shm/psm_*，worker 又必须能读取它交付的共享
内存。直接挂载主机共享内存会影响其他项目，完全独立的 worker 共享内存
则破坏原生加载路径。本次使用**本项目 storage 容器的私有 IPC/shm**，worker
通过 `--ipc=container:<owned-storage-id>` 加入；不挂载 host /dev/shm。

单机部署选择 host network 下的项目专用 loopback 副本地址，避免额外网络
代理或修改已有 Kubernetes/CNI。只增加自己拥有的本地监听端口；没有修改
主机接口、路由、防火墙、已有监听服务或原集群。它不是内核级网络隔离，不能
如此宣称；文件、共享内存、进程、权限和资源限制分别核验。loopback 地址是
副本地址，**不是新的物理节点**；D480 的一节点四卡 inventory 保持不变。

依据本机 Podman3.4.4 的[IPC文档](https://docs.podman.io/en/v3.4.4/markdown/podman-run.1.html#ipc-mode)，
但是否实际生效以容器 PID、namespace inode、cgroup 及实际读写见证为准。
没有安装软件、构建镜像、加载模型或触碰实验远端。

## 实际容器见证

复用 D477 已保留的官方 Ubuntu22.04 镜像：
`e8df6c74f650ad385efe2e20103eca9b0ad50690fb3c63d49fa00762412bbf55`。
辅助使用本机静态 BusyBox（SHA
`b3c1009e1b5c927e537487c80639cdf404f69e3eb49371d9be5d841672be3ff9`），
只读挂载，不生成新的镜像或模型数据。测试载荷为几十字节，与论文工作负载无关。

实际通过：

- storage 和 worker 处于同一个私有 IPC namespace，与主机不同；共享64 MiB
  tmpfs，不暴露主机测试 sentinel，也不出现 GPU device。
- worker 读取并校验 storage 创建的对象，然后只删除该测试对象；storage
  随后确认删除。主机自己创建的 sentinel 内容/inode 未变，并由本测试清理。
- 推理主机和 worker 分别向实际 storage loopback HTTP 端点读取正确内容。
- 两个真实进程启动在事先受限的父组内，逐容器 memory.max=128 MiB、swap=0、
  pids.max=128，实际 affinity=2,26；父组 high384/max512 MiB、swap0。
- 父组峰值233,533,440 B；high/max/OOM/swap全0。两个容器、此次专用 pause
  均已按标签/PID出生身份关闭；原scope实际递归成员为空后关闭。

这不覆盖 CUDA、LoRA数值、native large shm loader、取消/并发全生命周期或
完整性能。Podman pause先由其自身scope产生，再在任何工作容器之前纳入父组；
不能把这个事实写成 pause 从出生即处于共同父组。

首次容器测试退出127：辅助 BusyBox 被挂成非标准 multicall 文件名，日志为
`primelora-probe-busybox: applet not found`。已由该容器完整ID匹配的journal取回
原始错误；不是 HydraServe 错误。原退出状态、代码、日志和关闭回执保留。
仅修正辅助文件名后通过；没有为通过而改变内存、端口、IPC或其他安全条件。

## 原生连接为什么还需绑定来源地址

固定官方 commit `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b` 的
`vllm/executor/gpu_executor.py::init_connection` 使用
`ip_list.index(address[0])` 识别对端。真实内核小测确认：连接到另一个
loopback别名时，若不绑定本地来源，来源为127.0.0.1，而不是分配给副本的别名。
因此仅换成 host networking 会破坏原生副本身份，不能用“端口可访问”证明兼容。

新增 `HydraServe_project/runtime/loopback_transport.py`，生成未安装的源代码
overlay。它完整SHA固定四个输入文件，仅做以下 opt-in 必要调整：

1. app HTTP、local storage、backbone provider 的监听地址改为可配置；未开启
   时仍是原值0.0.0.0。
2. 原生三个主动 TCP 连接点，在开启 HYDRASERVE_BIND_SOURCE_IP 时明确绑定
   当前 ip_list[my_rank]，再执行原连接流程。端口、消息、rank选择、pipeline
   拓扑、KV迁移、合并策略和tensor计算不变。
3. 本地平台环境仅接受规范的IPv4 loopback，并配置原生VLLM_HOST_IP；
   地址合法性不是占用证明，真正创建时仍须检测端口、所有权和实际监听。

AST 逆变换核验：去除新增可选来源绑定，并还原监听默认表达式后，四个文件的
AST 与固定官方源码完全相同。原官方 checkout 未改动。

实测执行这个源代码版本的原生 `init_connection`，建立两个 stage 的 invocation
与communication连接：双方 peer/local 地址正确，双方可双向交换测试载荷，
原 model.set_socket 路径实际被调用。模型对象为惰性替身，没有CUDA或GPU计算。
测试只覆盖初始双stage连接，**没有证明 consolidation/KV迁移或GPU pipeline正确**。

5项新测试加此前27项，共32项通过（0.952s unittest时间）；资源峰值27,045,888 B，
全部内存事件/swap为0，实际scope关闭。完整source SHA、输入漂移拒绝、原生
代码保真、地址校验、真实默认来源和双stage传输均有独立测试。

## 下一步及仍未完成范围

将D480平台接口连接到上述真实生命周期组件，仍需实际官方后端镜像；D479同一
下载正在推进，不重启、不重复已经校验的blob。随后必须验证实际安装源码、
库/依赖、原生模型加载、D479 LoRA shared-owner绑定、请求身份/固定输出/
原生token计时及100→500-ID→1000→4000资格。

原backbone provider整模型CPU物化和复制可能触及内存包络，尚未改造或资格；
唯一模型、LoRA、trace保持不变。网络地址与IPC小测不能代替这些剩余项，也不能
以普通vLLM替代原生后端。当前没有新增性能结果，没有新增PrimeLoRA候选。

原始证据：实验仓库 `results/ieee_tc/hydraserve_qualification/d481_20261013/`。
curated证据：本仓库 `paper_results/ieee_tc/hydraserve/d481/`。失败与修正分别保存，
运行中的D479下载日志/可变缓存、凭据和任意镜像环境不纳入封存证据。
