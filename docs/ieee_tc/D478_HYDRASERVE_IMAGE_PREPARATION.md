# D478：HydraServe 官方运行环境准备（下载与续传）

2026-10-12。用户要求先完成 HydraServe 适配和实际结果，再比较全部基线、
优化 PrimeLoRA。本次不启动新的 PrimeLoRA 候选，不重跑已封存基线，不使用
与本项目无关的现有 k3s。完整 PLAN、当前状态及最新两份基线规则已重新核查。

## 本轮已确认的进展

- 复用 D477 的实际 rootless GPU 容器路径，不改全局容器配置或驱动。
- 官方 README 指定的后端镜像可匿名读取；不是因为后端版本旧而排除它。
- 固定镜像为
  `registry.us-east-1.aliyuncs.com/kubernetes-fc/modelscope-vllm@sha256:7a5c182f2296dcafacf9576cf09c3d4d188eda4738d51c081a7ca009f0c4a32f`。
  配置摘要 `sha256:b36aba10152151e75421368ee4abe7dd445fc0cef3c86337a4a18781db7ddfe9`；
  Linux/amd64，镜像记录创建时间 2024-12-03，79 个层条目，压缩层合计
  4,049,702,610 bytes。镜像创建时间不等于论文发表时间或源码 commit 身份；
  下载后仍须核验实际后端及其与已保留源码的差异。
- 新建项目私有 overlay 存储，使用已安装的 fuse-overlayfs 1.7.1；避免对
  79 层使用 vfs 的重复层复制。不修改 D477 的已封存镜像库及证据。
- 实际 `podman info` 成功。项目私有 pause 进程核对 PID/birth/UID、私有 pidfile、
  cgroup 和 InvocationID 后，在下载或模型工作开始前迁入既有受限父组；原空
  pause scope 已关闭。仅说明这个辅助进程的管理路径可行，不冒称全服务或
  Kubernetes 已合格；不隐瞒 pause 最初出生在独立用户 scope。
- 首次下载在读取未创建的匿名 auth JSON 时停止，未传输镜像；失败保留。
  修正只创建空 `auths` 文件，复用同一已确认的存储路径，再进行唯一实际下载。
  第一 scope 已按准确身份和真实递归空进程检查关闭，峰值 92,708,864 bytes，
  high/max/OOM/swap 均为 0。

## 首次实际下载与资源边界

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d478_20261012/`。
首次实际下载在 `pull_attempt/`，历史 tmux `tc-d478-hydra-pull`；父 scope
`primelora-tc-aux-d4780000000000000000000000000002.scope`，InvocationID
`66c21cb6aba94dfab90477d974ae5132`，inode `23596693`。

内存 high/max=3/4 GiB、swap=0、CPU=2,3,26,27；无模型或 GPU 容器启动。
启动前按 30 GiB 新增峰值检查了原有磁盘准入，150/100 GiB 门槛不变。
下载监视 100 GiB 运行磁盘下限及 16 GiB 主机可用内存下限。
当前不是推理性能实验，下载软件的时间不计入服务延迟或物理 GPU 生命周期。

官方仓库传输速度较慢，下载子命令有 1,800 秒的原始墙钟保护。若超时，
保留失败，不直接重下整个镜像。对已有临时压缩文件建立了同文件系统硬链接，
只保留 inode、不复制数据，也不改变正在执行的下载；它们仍是可变缓存，
不作为封存内容证据。后续须依据压缩 blob SHA、长度及服务端 Range 语义验证
后才可复用/续传；不能把未完成的文件登记为镜像已下载。成功后同样先核验
镜像身份，再清理本项目明确拥有的多余下载缓存。

## 23:31 CST 后的实际终态与缓存核验

原下载连接已结束，镜像未完整取得。收尾中 `owned_pids` 遇到进程退出时的
成员变化，原异常和最终结果尚未写出即停止；traceback 原样保留。不伪造原
return code 或 `image_result.json`。结束时机符合原 1,800 秒上限，但原被捕获
异常未序列化，因此将“下载超时”记为推断，而不是恢复出的原始异常字段。
外部核查实际递归进程列表为空、InvocationID/inode 一致后关闭 scope。
峰值 1,520,467,968 B；high/max/OOM/swap 均为零。没有模型或性能实验。

- 保留压缩字节共 1,172,708,833 B。79 个层条目实际对应 66 个独立 blob；
  其中 64 个完整摘要和长度已核验，共 233,300,347 B。
- 两个未完成 blob 分别保留 36,355,620 B 和 903,052,834 B，预期完整长度
  为 189,594,758 B 和 3,626,807,089 B。部分文件不是合格镜像层。
- 下载期间的新连接出现 TLS 错误；当前进程与原下载的八个代理变量、网络
  namespace 相同，不能归因于代理变量遗漏。单独直连检查超时，未改全局网络。
- 原连接停止后，再次读取同一最大 blob 的头部和 1 MiB 偏移处，均得到准确
  HTTP 206/Content-Range，内容与保留文件一致。支持针对这份缓存继续取缺失
  字节；仍须最终完整 SHA，不把两个片段相同当作完整内容认证。
- 首次离线检查错误地比较整个 stat，包括读取可更新的 atime；保留失败，
  改为比较 inode/device/size/mtime/ctime 后完成核验。没有更改镜像字节。

复用既有无损压缩工具，只扩展为接受明确指定的已结束实验 final monitor 日志。
8 项测试通过；16 个已确认释放的历史日志经两次解压 SHA 核验，释放
1,679,425,536 B，全部原内容及运行身份保留。147 项保护清单零变化。
完成时可用 161,357,086,720 B，恢复 150 GiB 新任务门槛。该操作不是实验结果。

续传入口为 RAW/resume_image_blobs.py、tmux `tc-d478-hydra-resume`，scope9。
仅续传两个缺失 blob，不再次拉取 64 个完整层，不导入或解压整个镜像，不启动
推理。每个部分文件在追加前重新核查远端两处样本、长度、inode 和 HTTP Range；
完成后验证全文件 SHA。下载保护改为 3 小时，依据剩余字节与实测下载速度；
这是软件获取边界，不改变任何推理超时、SLO 或 baseline 配置。原失败完整保留。
磁盘 150/100 GiB、内存保护和专属 CPU/资源域不变。续传结果尚待确认。

续传依据：[Registry V2 分段读取规范](https://distribution.github.io/distribution/spec/api/#fetch-blob-part)。

## 原生适配的两项新源码边界

1. `src/utils.py:create_deployment` 实际仅申请 `aliyun.com/gpu-mem` 和主机
   memory；传入的 GPU core percentage 未写入资源限制。不能在本机适配中
   额外强加一套原实现未执行的算力限额，也不能冒称已复现了算力隔离。
2. `storage_server/local_server.py` 启动时有通配清理 `/dev/shm/psm_*` 和
   无界分配重试。**没有执行该脚本**。真实部署必须使用项目独占共享内存空间
   或精确所有权回收及共同资源保护，不允许清理其他程序的共享对象。

原生请求调度、准备重叠、pipeline/consolidation 及修改版 vLLM 保留。
尚未选择或实现完整 Kubernetes 替代层；必要的生命周期移植必须映射真实
容器、节点、GPU、地址和回收，不能伪造 Kubernetes ready 响应或多计算节点。
一个计算节点四张 GPU 不自动成为四个物理节点。

## 下一步与不能得出的结论

先完成并核验该次下载，确认官方镜像实际源码/依赖/本机运行条件，再验证
模型加载和 D447 已定位的全局层坐标、共享层及合并目标 LoRA 绑定。
Kubernetes/完整平台、CUDA/模型、LoRA 数值和完整负载资格均尚未完成；
本记录不是 HydraServe 性能行。3B、W1/W2/G2、消融和敏感性均未取消。

依据：[HydraServe 官方实现](https://github.com/LLMServe/hydraserve)、
[Podman 3.4.4 存储及私有路径说明](https://docs.podman.io/en/v3.4.4/markdown/podman.1.html)、
[kind rootless 所需 CPU 委派](https://kind.sigs.k8s.io/docs/user/rootless/)
（本轮核查）。镜像可访问性、摘要和资源状态来自本机实测，不由文档推断。
