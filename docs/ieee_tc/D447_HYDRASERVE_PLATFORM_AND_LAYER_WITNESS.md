# D447：HydraServe 平台边界与真实 7B 元数据层映射见证

2026-10-12。主线为已完成五系统7B W0后的HydraServe资格，不是新的性能运行。
完整PLAN、当前状态、最新baseline规则已核对。复用D78/D80一次性只读交付缓存；
没有重新发布、下载权重、生成负载、建立解压池或修改远端服务。

## 已经完成什么

- 实际检查了用户级Podman、CDI规格、Kubernetes访问条件和资源委派，不再沿用
  “本机完全没有容器环境”的历史概括。
- 使用既有7B基座的两个safetensors头、finance_lora/medical_lora的头，执行
  固定官方源码中的分层算法、名称解析、按名查找和共享层方法；不导入torch，
  不调用模型下载或具有改写风险的partition函数。
- 新增一个窄范围CPU诊断及5项测试，复用现有Loquetier的安全头读取函数及
  5项测试，合计10项通过；没有重建实验框架、修改基线算法或搜索性能参数。
- 147项旧结果/受保护文件零变化，原生HydraServe checkout保持干净。

## 1. 平台实际观察，不等同于GPU系统资格

05:42–后续复核的观察记录在D447原始目录的platform_probe.txt和platform_detail.txt。

| 条件 | 本次实际观察 | 能得出的结论 |
|---|---|---|
| 用户级容器 | Podman3.4.4、rootless、cgroup v2、systemd/crun、overlay；镜像/容器均0 | 用户级容器元数据接口可用；未启动GPU容器 |
| GPU规格 | nvidia-ctk列出9个CDI名字：4索引、4UUID、all | 存在设备规格；不证明容器设备注入成功 |
| NVIDIA runtime | 可执行的nvidia-container-runtime存在；所查两处OCI hook目录不存在 | 有进一步限定测试的候选路径，未改全局配置 |
| Kubernetes | k3s v1.36.4+k3s1，root拥有的k3s-server服务active | 集群存在，归属与可用访问范围尚未核定 |
| 当前用户访问 | 无默认~/.kube/config；/etc/rancher/k3s不可读；kubectl/helm为root0700；Docker socket不可读 | 不能以当前凭据启动官方Kubernetes路径 |
| 用户资源委派 | DelegateControllers仅memory、pids | 现有CPU亲和性与内存scope可用于本次小检查；不能据此声称rootless kubelet/Pods已满足共同资源合同 |

[NVIDIA官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
将Podman原生CDI设备参数支持起点列为4.1，并描述了非CDI runtime的注入方式。
因此3.4.4不能直接按新版本示例使用，也不能仅凭版本判定GPU容器绝对不可用。
本次未使用任何替代注入或更改runtime。

[K3s官方rootless说明](https://docs.k3s.io/advanced#running-rootless-servers-experimental)
要求适当的cgroup委派；当前只是先决条件检查，没有启动另一集群来规避现有边界。
已向用户询问现有集群归属和可用kubeconfig路径，不要求粘贴凭据。
不读root凭据、不改未知集群、全局网络、驱动、权限或既有服务。

## 2. 真实7B元数据驱动的名称结果

固定官方HEAD为8ae605de354ccfa2e9095514cdb4a9e9c56aa56b；五个源码整文件SHA先验证，
随后仅提取指定的未修改函数体执行。注解延迟求值以免导入GPU模块，函数体不改。
[HydraServe官方实现](https://github.com/LLMServe/hydraserve)的pipeline、合并、准备重叠
以及专用vLLM0.4.2均未替换。

既有基座头含323个tensor记录，其中32个F32 rotary inv_freq由原生getType分类为3，
不进入分层容量评分；其余291个为F16。首次CPU检查错误地要求全部F16而停止，
该失败已保留；修正的是诊断的元数据假设，未改变模型或任何baseline运行参数。

官方算法给出的累计层边界如下；这是功能见证，非pipeline性能配置搜索：

| 分段数 | 全局层边界 | 每个被测adapter中，直接按原始名字查找时的错误全局身份数 |
|---:|---|---|
| 1 | 0–32 | 0/128 |
| 2 | 0–16、16–32 | 第一段0/64，第二段64/64 |
| 3 | 0–11、11–22、22–32 | 0/44、44/44、40/40 |
| 4 | 0–8、8–16、16–24、24–32 | 0/32、32/32、32/32、32/32 |

两个adapter各有128个A/B齐全的目标模块。见证将每个全局模块名作为不含tensor的
身份标记，通过原生LoRAModel.get_lora查找，验证的是**名字所指的全局层**。
例如第二段model.layers.0应使用全局16，而原始完整字典中的同名项属于全局0。
由同一个官方transform导出的只读名称视图可以使这项身份检查全部一致。

边界：尚未将名称视图接入实际LoRA loader/manager，不代表GPU执行、数值正确、
实际张量指针或请求正确性已经合格；头SHA不是完整tensor内容SHA。没有用零权重
输出相同来替代绑定证明，也没有重新扫描两模型整池。

## 3. 合并的新增执行证据

在2→1、原rank1的派生映射上，执行原生check_duplicate，使用带全局层身份的
惰性Python对象代替torch decoder，确认16个目标decoder引用原来的对象本身。
这把先前源码推理变成了可执行的对象身份见证，但不证明GPU并发安全或LoRA slot
正确性。不能据此断言已经观测到错误数值或数据竞争。

下一步必要适配必须分别保持：

1. 每个初始stage的全局→局部LoRA坐标正确，权重/rank/alpha不变。
2. 目标新增层有正确绑定，复用层不被另一manager破坏slot/batch mapping。
3. 迁移和use_dest切换前后同请求仍使用同adapter；原pipeline/合并/调度不替换。
4. 所有真实workers满足共同内存/CPU/swap/物理GPU计量，再进入原定资格回放。

没有在平台资格尚不明确时安装依赖、拉取大镜像、生成模型分片或编译后端。
也没有把普通vLLM、关闭合并后的服务或自制调度器冒称HydraServe。

## 4. 资源、保全及来源

CPU检查隐藏GPU，CPU2,3,26,27、MemoryHigh3GiB/Max4GiB/swap0；六个owned scope
均核对InvocationID和真实递归cgroup.procs为空后停止。high/max/OOM/swap增量均0。
平台scope峰值99,512,320B；首次检查11,366,400B；最终测试11,571,200B；
独立见证10,682,368B；保护核查11,649,024B；14项来源SHA复核6,131,712B。
最后GPU均15MiB背景显存、利用率0；磁盘可用162,472,525,824B，约151.32GiB。
这不是下一重型任务的准入保证；仍须运行前重新检查完整门槛。

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d447_20261012/`。

- layer_identity.json：769c8228c611ba12506bc37cbfc13ebcb4ae40ecd89d8ac28bc8479b3c50b808。
- CPU测试：10通过，0.073s；先前0测试的头格式检查失败单独保留，不是性能重跑。
- 窄诊断复用Loquetier已有_tensor_header，不改原读取器、模型、trace或生成合同。
- 新程序/测试与原始记录SHA见paper_results/ieee_tc/hydraserve/d447_cpu_platform_summary.json。

## 5. 当前结论及主线

HydraServe仍是待资格系统，不进入已完成五系统W0主表。新结果证明需要具体的
层坐标和共享对象适配，而非以旧后端版本、没有一个adapter字段等理由直接排除。
平台访问待补充信息；这不是整个实验目标被阻塞或取消其余任务。
后续继续剩余7B/ElasticLocality与W1/W2证据、必要PrimeLoRA优化，再推进3B、
核心消融、机制/敏感性与审稿文档。已封存五系统W0不重复，未到目标前不画图。
