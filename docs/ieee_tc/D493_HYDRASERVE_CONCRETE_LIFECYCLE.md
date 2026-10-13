# D492–D493：独立生命周期接口与原生私有存储实际接通

2026-10-13。必要平台适配，不是 HydraServe 性能优化或完整复现。
用户已确认现有 k3s 属于另一项目；本次未访问或修改其服务、权限、配置及资源。

## 实现与实际验证

新增 `HydraServe_project/runtime/isolated_platform.py`，实际实现 D480 的
LifecyclePlatform，而不构造 Kubernetes Pod/Ready 替身。复用 D477/D481/D491
的 rootless Podman、父资源域、私有 IPC、loopback 地址与 GPU UUID 选择方式。
控制器依旧执行原生 claim、best-fit、create_pods、initialize_pods、release_resource。
一个物理主机保留一个 inventory 条目，四卡不是四个节点。

每个容器先执行不导入 torch 的入口，在主机核查实际 PID、出生身份、CPU affinity、
父组身份、子组限制及 IPC 后才获准执行其原生命令。这样不会在原生 app 的
异步加载前增加一次 torch 导入，破坏原生启动重叠。地址返回不是模型就绪；
原 ResourceManager 仍等待 `/init`。删除确认核查实际进程及递归 cgroup 成员，
不凭 CLI 返回或虚拟文件大小恢复资源。创建失败不隐式重试。

依据[固定官方源码](https://github.com/LLMServe/HydraServe/tree/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b)，
本轮重新从官方 raw URL 取得 resource_manager.py，SHA 与本地固定输入一致：
`3a88b5f784eb089fd31b7378f688c356b5c57cad837ba1f334c01b4f8ab56d52`。
浏览工具当时返回 429，未将该错误当作论文证据；实际联网源码验证成功。
原生 local_server 仅使用此前 D481 已核验的监听地址 overlay，缓存、双线程
下载、共享内存协议不变；官方 checkout 仍干净。

D493 实际通过：

- 真正运行原生 local_server，分配并触碰 1 GiB 私有共享内存，原生 health 与
  free 命令正常；不是自制存储模拟服务。
- 原生 ResourceManager 创建两个不同容器、选择 GPU 0/1，两个进程均能读取
  该私有对象并通过原生 health 协议访问存储；主机共享内存不被暴露。
- `/init` 使用明确标记的**无模型测试端点**，延迟确认期间原生控制器确实等待；
  两次删除均完成实际退出后才恢复原生资源账本。它不证明模型已经 Ready。
- 私有 GPU device 可见性按指定容器记录。CDI runtime 将容器内部
  NVIDIA_VISIBLE_DEVICES 改为 void，但实际 GPU device 文件被注入；不能据
  该环境字符串推断无 GPU，也不把本轮无 CUDA 操作称为 GPU 数值验证。
- 两个 worker、storage、所属 pause 均释放，scope 实际递归为空后关闭；
  parent high3/max4 GiB、swap0、CPU2,3,26,27，child max2 GiB、swap0。
  峰值1,450,344,448 B；high/max/OOM/OOMkill/swap均0。

这仅为功能检查包络，不能用作主性能比较。D492 首次 launcher 在任何容器创建
前因两个仓库的 scripts 包重名而导入失败；D493 先导入主仓库 preflight 后再
加入 baseline 路径。原错误/代码/实际空组关闭证据保留，没有模型失败或 OOM。

实际运行后增加两个并发安全收敛：入口记录用原子发布避免半写 JSON；取消时
等待有界容器命令结束后再清理，避免 create 与 cleanup 竞争。后者有实际短
子进程取消测试。D493 运行时的两个原文件另存 `*_executed.py`，不把后续改动
追写为当时已执行；更新版通过同组 CPU 回归，尚未重新进行完整模型 qualification。

## 证据边界与下一步

未执行 backbone provider→local shared memory→C++ fastload→app/init 的完整
模型链路，也未完成原生请求 LoRA 身份/固定输出/token 计时接入。D491 的分段
模型/共享 LoRA GPU 数值证据继续复用，不重复安装、构建或部件 CUDA 小测。

下一步直接复用本接口接入原生 provider/fastload 和 worker app。已有 backbone
是两个 HF safetensors 分片，native provider 的离线分支只读 MODEL_PATH 一个
文件，须以只读分片 iterator 适配，不能执行会删除/重写唯一模型的 downloader。
只生成小型配置/分段元数据及链接，禁止复制整套权重。provider 默认 model_set
包含 13B，必须显式注册现有 7B/3B，保留实际 CPU packing、分段与线上序列化。
LOCAL_SERVER=1/FAST_COLDSTART=1 下 ENABLE_PARA=0 仍是原生共享内存快路径，
不能误作普通 loader 替代。私有存储容量必须按完整权重字节和共同80 GiB包络
审计；本轮 1 GiB 仅测试原生存储生命周期，不是完整模型配置。

然后按100请求→500-ID→1000→4000得到 HydraServe 实际结果，再作完整可用基线
比较和 PrimeLoRA 优化。W1/W2/G2、3B/B6、A1–A5、S1–S13、成本/文档/最终图表
全部保留。本轮没有 TTFT、TPOT、SLO 或 GPU-s 性能结论，不增加排名行。

原始证据：原实验仓库 `results/ieee_tc/hydraserve_qualification/d492_20261013/`
与 `d493_20261013/`。精选小型成功/失败证据与逐文件 SHA 存本仓库
`paper_results/ieee_tc/hydraserve/d493/`。
