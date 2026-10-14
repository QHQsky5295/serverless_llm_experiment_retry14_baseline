# D522：原生设备元数据、非零 LoRA 与 CUDA Graph 对照

2026-10-14。承接 D520/D521，同一优化假设：当前 CPU 已知的请求—槽位映射
可以直接构造原生元数据，避免先传入 GPU 再归约回 CPU 的同步依赖。本轮没有
改变九公式、路由、驻留、准入、并发或评价阈值；也没有新的性能比较结果。

## 检查与结果

复用现有 preflight/资源保护入口，在实际 vLLM0.30.0、Torch2.13.0+cu130、
CUDA GPU0 上执行。沿用 D476 的容量1024行、14槽；有界非零算术 fixture
只在内存中存在，不生成权重、工件池或执行负载。原生和候选调用同一
`PunicaWrapperGPU.add_lora_linear`。rank16、17个真实行，FP16非零输入/矩阵。

| 检查 | 观测 | 范围 |
|---|---|---|
| 原生整数元数据 | 10/10完全一致 | 单adapter、混合base、槽位重排、全14槽、全base；specialization开/关 |
| 非零LoRA算术 | 8/8非零且逐位一致 | 其余2个全base负对照输出严格为零 |
| 原有CUDA Graph | 10/10逐位一致 | 先以原生路径capture，再接入候选；不重建GPU目标 |
| 连续异步更新/graph replay | 两组各32次全部一致 | 中间无CPU readback/barrier；银行仍执行确切复制事件退休检查 |
| 固定pinned payload | 每publisher99,040B | allocator实际增量同为99,040B；不是模型服务总内存 |
| 额外回归 | 521通过，0失败/错误/跳过 | 现有worker/source观测、引用与smoke；CPU fixture不冒称GPU证据 |

实际reuse_waits=0，说明本次没有观察到必须等待的银行重用；不能把它解释为
事件等待分支已在设备上触发。该分支已有D521所有权单元测试。元数据与算术
检查使用相同原生算子，证明此有界条件下构造替换等价，不是独立数学正确性
证明，更不覆盖整个模型或全4000请求。`n_correct`继续为未知。

第一次启动在导入项目模块时失败，CUDA检查用例为空。修正的是既有preflight
辅助入口缺失的项目导入路径；第二次使用不同文件/执行键，原记录全部保留。
它不是候选算法数值失败，也不能从有效结果中删除这次启动记录。

两次设备运行服务域均已消失，外置域均核对精确身份及递归进程为空后关闭。
CPU回归实际峰值1,059,352,576B，high/max/OOM/OOMkill/swap均0，关闭路径消失。
未操作远端服务或共享k3s，未重跑HydraServe，未覆盖旧结果。

## 完整服务资格仍待验证

为避免把孤立算术当作实际安装，下一步D523复用原100请求Full运行。新增配置
只开启 `ieee_lora_metadata_policy=cpu_live_mapping_v1`；与D475配置逐字段比对，
除唯一策略及新输出/临时路径外相同。使用Full的全局HOST预留和原生预算attach，
不会在孤立preflight中绕过这层所有权。现有source/worker观测增加只读publisher
计数，不添加RPC、设备屏障或额外按请求采样。

资格要求：所有100请求原生完成/目标数/adapter绑定合格；实际安装成功、
新增pinned被原预算计入、计数证明被调用、无错误回退、生命周期释放完整。
不以100条延迟判定优化成功。通过后一次完整4000 W0，才用冻结V1/单轮V2
评估G1；W1/W2/G2、3B及完整后续矩阵继续保留。

RAW `results/ieee_tc/p2_backend_qualification/d522_20261014/`；curated
`paper_results/ieee_tc/p2_backend/d522_metadata_device/`。D523尚未启动。

依据：使用vLLM skill核查原生执行边界，复用run-experiment入口和github-sync
备份检查；通用skill示例阈值不替换本项目冻结协议。
[vLLM原生metadata实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/punica_wrapper/punica_gpu.py)
明确包含同步的元数据准备路径；
[CUDA事件语义](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)
用于核对源银行的事件生命周期。本机实测而非网页版本推定CUDA行为。
