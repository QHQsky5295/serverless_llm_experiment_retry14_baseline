# D477：独立 GPU 容器已验证；HydraServe 原生资格仍未完成

2026-10-12。遵循用户最新顺序：先完成 HydraServe 必要适配及实际结果，再作完整
基线比较和 PrimeLoRA 优化。原有 k3s 不属于本项目，本次没有访问或修改其配置、
权限、服务、网络及数据。没有运行新的模型回放、修改基线算法或搜索性能配置。

## 1. 实际得到的证据

在项目独立 Podman 存储中，仅下载一次官方 Ubuntu 22.04 小镜像；未下载模型，
未建立第二套 LoRA 工件池。固定镜像身份：

- image ID：`e8df6c74f650ad385efe2e20103eca9b0ad50690fb3c63d49fa00762412bbf55`。
- image digest：`sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401`。
- image size：80,652,148 B；镜像及后续尝试复用同一存储，不重复拉取。
- Podman 3.4.4、NVIDIA runtime 1.19.0、runc 1.3.3；没有升级宿主软件或驱动。

最终容器实际执行 `nvidia-smi -L`，只列出指定 GPU：
`GPU-8f51f82e-dce8-22cf-1693-e62b8464f946`。进程身份从宿主 `/proc` 验证：

- 位于预先建立的本次受限父资源域内；
- CPU 亲和性为 `2,3,26,27`，在容器入口设置后才执行检查；
- 实际子组 `memory.max=134217728`、`memory.swap.max=0`、`pids.max=128`；
- 父组 MemoryHigh 3 GiB、MemoryMax 4 GiB、swap 0；
- 无网络、只读根、删除 capabilities、no-new-privileges，不使用 privileged；
- 命令退出码 0，指定容器已删除；捕获的资源域和专用 pause 进程均核验身份后关闭。

最终父组峰值 141,168,640 B，high/max/OOM/swap 均为 0。这是 **NVML 设备访问和
小型进程约束见证**，不是 CUDA 张量运算、模型数值、LoRA、Kubernetes 或 HydraServe
性能验证。没有用它补齐主表的性能值。

## 2. 保留的失败和必要修正

| 尝试 | 实际结果 | 处理与边界 |
|---|---|---|
| direct | Docker Hub 直连超时，未创建容器 | 启动器遗漏本机已有代理；后续仅继承已有代理，不打印凭据，不换不可信镜像源 |
| proxy | 镜像获取成功；旧 Podman 拒绝 NVIDIA runtime 的 NoCgroups 模式 | 不绕过能力校验；改用真实、预先委派的 memory/pids 子组 |
| delegated | Podman 拒绝超过 50 字符的 runroot | 保留失败；复用已获取镜像及原短 runroot |
| parent | 容器实际启动、内存父组正确，但 worker CPU 亲和性被 runtime 重置为全部 48 核 | 当作约束失败，停止并删除确切容器；不视为合格运行 |
| affinity | 容器入口先绑定 CPU，再执行 NVML；实际父组、子组限制、GPU UUID 和退出均通过 | 仅接受上述窄范围平台证据 |

这些不是重复性能试验或寻找更好结果；失败发生在不同的启动/约束环节，全部保留。
未改变原工作负载、SLO、GPU 计量、真实远端交付或原生 HydraServe 核心机制。

## 3. 仍需解决的边界

1. 用户级控制器实际只有 `memory pids`，无 `cpu/cpuset`；只设置 Delegate=yes
   不能制造祖先未委派的控制器。官方 rootless kind 明确要求 CPU 委派；当前
   无免密码本机管理员权限。没有降低主机保护要求或修改 user@ 系统服务。
2. Podman 自动把专用 pause 进程放到独立用户 scope；本次已通过私有 pidfile、
   UID、出生时间、InvocationID 和真实成员确认后释放。它没有处于本次受限父组，
   所以**整个服务资源域尚未合格**，不能只靠 worker 在组内便声称完整安全资格。
3. 后续须比较官方原生编排的独立部署条件与必要的非云平台接口适配：只能替换
   生命周期/资源发现的实际平台接口，不能伪造 Kubernetes 返回值、模拟 ready，
   或替换原 scheduler、pipeline、consolidation、启动重叠及修改版 vLLM。
   是否可以在这些边界内不依赖既有集群，仍需逐调用验证，尚未实施或宣布成功。
4. D447 的 stage-local/global LoRA 名称映射和合并时共享/新增层 manager 绑定仍
   需原生 GPU 数值见证。容器看得到设备不能证明这些问题已经解决。

若忠实执行确需本机管理员提供新的资源委派，须明确提出最小管理员协调事项；
不使用远端凭据猜测本机密码，不改动与本项目无关的集群，不关闭核心机制冒名。

## 4. 保全与下一步

先无损压缩三份已结束历史失败日志，复用既有显式路径归档器；两次解压 SHA
一致，失败记录没有改写，共释放 188,305,408 B。receipt 在主仓库
`paper_results/ieee_tc/storage/20261012_d477_closed_outcomes_lossless.json`。
推理机 150 GiB 启动/100 GiB 运行底线保持不变；小镜像之后剩余约 150.06 GiB，
不足以直接承诺大型后端构建空间，下一重型任务必须重新测量增长与容量。

下一主线仍是 HydraServe：实际平台接口、原生后端与 LoRA/合并资格，然后
100 → 500-ID → 1000 → 4000 的既定流程。已封存基线不重复，PrimeLoRA 新优化
暂不启动。7B W1/W2/G2、3B、消融、机制和敏感性均未取消。

原始证据位于
`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d477_20261012/`。
来源清单、闭合回执与窄范围验收保存到 `paper_results/ieee_tc/hydraserve/d477/`。

依据：[NVIDIA CDI 非原生 runtime 支持](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)、
[Podman 3.4.4](https://docs.podman.io/en/v3.4.4/markdown/podman-run.1.html)、
[kind rootless 条件](https://kind.sigs.k8s.io/docs/user/rootless/)、
[官方 HydraServe](https://github.com/LLMServe/hydraserve)。2026-10-12 核查；宿主
实际版本/结果优先于最新文档中的默认行为，不按新文档推定旧程序已支持。
