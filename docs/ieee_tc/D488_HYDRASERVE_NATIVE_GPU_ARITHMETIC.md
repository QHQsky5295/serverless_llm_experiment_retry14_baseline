# D487/D488：独立容器中的原生 GPU 与非零 LoRA 数值核查

2026-10-13。主线仍为 HydraServe 实际结果 → 全部可用基线比较 →
PrimeLoRA 优化。本次是原生后端正确性资格，不是性能回放，不优化基线参数。
现有非项目 k3s 未访问或修改；不新增权重、负载、工件池或模型副本。

## 结果与证明范围

| 检查 | 观测结果 | 尚不能推出 |
|---|---|---|
| D487 原生 Punica 测试 | 固定官方测试函数体、FP16 4096×4096、rank 8，通过 | 所有内核形状或完整模型正确 |
| D487 SiLU、FlashAttention | 小型实际 GPU 计算与独立表达式对照通过 | 完整 prefill/decode 性能 |
| D488 既有非零 LoRA | 两套既有权重，32 层×4模块共128位置，全部通过 | 实际 worker 的 stage/shared-owner 绑定正确 |
| 数值误差 | 最大相对 L2 误差0.0002507982426（约0.0251%）；最大绝对误差3.21958214e-6 | 跨后端完整生成逐字一致 |
| 无 adapter 负对照 | 每个批次索引 -1 的行保持零更新 | 完整服务无 fallback |
| 清理与安全 | worker退出、容器删除、GPU释放、父组实际递归为空 | 整套 HydraServe 生命周期已经合格 |

后端是自己编译的固定 HydraServe commit
`8ae605de354ccfa2e9095514cdb4a9e9c56aa56b` 修改版vLLM0.4.2，
Torch2.3.0+cu121、FlashAttention2.5.8；不替换成普通vLLM。
既有官方Ubuntu22.04小镜像、D486环境、Python/CUDA运行库均只读挂载复用；
没有安装、编译、网络访问、远端服务操作或基座模型加载。

D488逐模块读取原safetensors，两个逻辑adapter交替选槽。参考值在CPU以
float32计算 `x @ A.T @ B.T * alpha/r`，原生GPU路径使用FP16输入/权重
及原有Punica实现。保留官方FP16 `rtol=atol=0.005`，并在运行前额外固定
`||native-reference||₂/||reference||₂ <= 0.005`，要求参考和实际更新非零。
因此不能用“权重全零所以输出相同”通过。全部128位置和失败原始记录均保留，
未按误差选择模块或放宽阈值；这不是新生成的服务负载。

## 更正并保留本次校验错误

D487的原生三个GPU检查已经通过，随后完整权重SHA断言失败。原因是本次
检查脚本误将D447的 `header_sha256` 当成完整文件SHA；并非权重改变。
原始失败代码、traceback、容器退出码1、数值检查部分结果全部保留。
两工件的完整SHA与既有20260927内容清单一致，且彼此不同：

- finance：`971901c3140f83f57cbb19fcc86d33b4d010f6d8fc15fbec621f0347bc31a258`
- medical：`d2b1dcfe42193917d0b2e0f728e735f24ad119dd2c77cb9b20c983c7f688fa5c`

两者头部相同只说明相同张量布局，不代表相同数值；先前同SHA的口头判断已
明确纠正。D488只修正这一校验依据并继续非零LoRA部分，不重跑已通过的
三个检查，不修改adapter、原生内核、数据或数值容限。

D487控制进程继承了不提供 `os.pidfd_open` 的conda Python，容器已退出/删除后
专属pause收尾失败。随后用系统 `/usr/bin/python3`，核对捕获的PID出生身份、
UID、原InvocationID和成员集合后，使用pidfd关闭该唯一pause；退出后的成员
采样遇到竞态，单独的真实递归空检查和原scope关闭实际通过。此辅助错误也
保留，不伪造D487缺失的顶层result/退出码。D488固定系统Python并在启动前
验证pidfd接口；实际正常完成清理。没有信号发送给其他项目进程。

## 实际隔离与资源

复用D481的真实生命周期、父子资源域、PID身份和清理方法；worker先等待主机
核验，再允许CUDA。GPU只暴露一个指定UUID；无网络、私有IPC/shm、只读根、
无新增capability，不挂载host /dev/shm，不访问现有集群。
这与[本机Podman3.4.4文档](https://docs.podman.io/en/v3.4.4/markdown/podman-run.1.html)
核对，生效与否以实际进程和cgroup读回为准。

父组high3/max4GiB、swap0；实际worker max2GiB、swap0、pids512；
CPU2,3,26,27。此为小型内核资格包络，不冒充正式服务72/80GiB包络。
容器外每秒检查主机16GiB底线、磁盘100GiB底线和资源事件，300秒有限保护。
pause由Podman产生后、任何worker之前纳入受限父组，不声称从出生就在父组。

| 尝试 | 父组峰值（B） | high/max/OOM/OOMkill/swap | 最终实际成员 |
|---|---:|---|---|
| D487 | 670785536 | 全0 | 递归空、已关闭 |
| D488 | 506236928 | 全0 | 递归空、已关闭 |

D488 PyTorch峰值allocated394240B、reserved2097152B仅是张量分配统计，
不是全部GPU占用或生命周期积分；本次不据此计算GPU-s/request。
完成时磁盘可用202018508800B，下一任务仍重新检查。

## 下一步，不把内核通过当作复现完成

1. 复用已通过的环境/内核，不重复安装、编译、NVML或这些数值小测。
2. 将D479真实层坐标和shared-owner适配接入原生模型加载/合并路径，核验
   stage、destination新增/共享层、LoRA active槽位及use_dest后的实际数值。
3. D480生命周期接口仍需接入具体独立运行平台，D481私有SHM和地址组件复用；
   原生backbone provider/fastload安全加载、请求adapter/fixed-output/native-token
   路径尚未闭口。不得以裸vLLM冒称HydraServe。
4. 之后按100→500-ID→1000→4000取得实际请求结果，再作全部可用基线比较。
   单机四卡不伪造多物理节点；不触发的pipeline/consolidation如实记录。

原始D487/D488目录均在实验仓库 `results/ieee_tc/hydraserve_qualification/`；
小型source/失败/成功/清理证据归档本仓库 `paper_results/ieee_tc/hydraserve/d488/`。
旧投稿147文件和用户修改继续保护。结果解释遵循analyze-results的“观测—
解释—边界—下一步”，不计算用户已取消的运行间CI，不生成性能图。
