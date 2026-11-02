# D1017：恢复启动路径遗漏的已保留 HOST 描述读取

2026-11-02，接续已完成的 D1016。上一轮为实质进展：当前 1000 请求原生
窗口已经完整测量和备份，不追加同前缀 profiling。主线仍为 3B PrimeLoRA
在已完成 baseline 适配/观测后的优化；完整范围不缩减。

## 新证据，而不是重复一个已关闭优化

逐请求检查实际 GPU 引用回执，得到：

| 来源 | 完整请求数 | Python 描述读取 | D747 融合描述读取 | 缺字段 |
|---|---:|---:|---:|---:|
| D1000 完整 W2 | 4000 | 4000 | 0 | 0 |
| D1016 attempt2 诊断 | 1000 | 1000 | 0 | 0 |

来源、字段、SHA 和提取身份见 `paper_results/ieee_tc/p2_backend/`
`d1017_restore_descriptor/entry.json`。计数只取每请求实际获取回执，不将
嵌套重复出现的 metadata 当作独立调用。最初误对 main_outcome 使用结果文件
的请求前缀得到零行；该探查未作为结论，改用相邻完整结果后人数精确匹配。

D747/D748 的融合实现逐 view 读取当前 Tensor/storage/pinning，减少接口
封装，不缓存状态。D749 完整 W0 从 D742 的 3727/4000 提高到 3787/4000，
资源从 14868.152527 降到 14847.164967 GPU-s；当时明确保留为后续父版。
D886 仍明确使用 D747 库，且旧 runner 设置其路径与 SHA。

当前 D1000 runner 没有这两个导出项；至少 D934 从 D886 配置转入新 runner
时已缺失（当前文件 SHA `86eb33ab6f59d95c9528149c0ce65d291626dd66c0038e52a8eafe76fe7662b9`）。
已查 D88x–D1016 记录，没有找到撤回这个逐 view 实现的决定。默认 loader
确实在未配置时选 Python，实际回执也证实了这一点。因此这是**已保留实现
未被当前启动配置启用**，不是证明 D747 算法失败后再试一次。D866 整图模式
与它不同，仍不启用；也不顺带恢复已关闭的 ISA-L 或其他参数。

## 唯一假设与公式边界

假设：当前同步控制仍消耗显著 CPU 时间，恢复已验证的实时描述读取可以
减少控制事务开销，并改善完整 W2 的共同 SLO。D1016 的窗口占比支持检查
这条路径，但不能预测改善幅度；旧 D749 的收益也不能移作当前结果。

九公式、tier 确认、实时 pinning、alias/storage 容量、allocator/workspace、
准入和所有权不变。没有删除检查、跨调用缓存、兜底回退或人为延迟；不改
到达预取、扩缩容、GPU 数、token 预算、热度、阈值或负载。库失败继续显式
报错。当前源码不改，复用 SHA 已核对的 D747 195688 B 库和 D1000 runner。

本轮联网复核 [vLLM 0.30 LoRAModel](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)
的共享 Tensor 语义、[PyTorch 2.13 pinning](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
的当前 storage 查询，以及 [EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的输入/执行边界。这些支持保留动态检查和减少接口调用的实现边界，不提供
本机加速百分比。vLLM 技能的通用性能数字/参数不替代本项目协议。

## 验证、执行与终止条件

直接复用 D747 的实际 CUDA 九状态、四种图及 D748 的原生构造/clone 资格。
本次只复核当前 loader 的九项既有配置/图接口测试，以及实际 ABI 能载入
同一库并立即读到同 Tensor 的 storage 变化。没有改 C++ 或生产算法，不
重建库、不新增模型 smoke、不重复组件性能微测。

通过后只执行一次完整 4000 请求 W2：新键
`d1017_20261102/d1017_3b_restored_fresh_descriptor_w2`。沿用 D1000 YAML，
由现有 runner 按新键覆盖 output/NVMe/HOST 三个路径；不创建新配置副本、
工件或 trace。没有 Python 栈/host-spans profiling，解压保持 stdlib。

```bash
env -u FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER \
  -u FAASLORA_D1000_PREFIX_COUNT -u FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT \
  -u FAASLORA_D1000_NATIVE_PROFILE \
  PYTHONPATH=/home/qhq/serverless_llm_experiment_retry14_baseline \
  FAASLORA_IEEE_HOST_INVENTORY_MODE=descriptor \
  FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so \
  FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044 \
  FAASLORA_D1000_RUN_ID=d1017_20261102 \
  FAASLORA_D1000_RUN_TAG=d1017_3b_restored_fresh_descriptor_w2 \
  FAASLORA_D1000_CONFIG=configs/ieee_tc/20261101_d1000_3b_route_confirmed_arrival_prefetch_w1.yaml \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

实际 worker 回执必须确认 `fresh_native_tensor_descriptor_v1` 和上述 SHA。
仅看到启动环境不算生效。完整结果按冻结 D648、V1/单轮 V2 比较 D1000
3690/4000、8008.311805 GPU-s；不以 D1016 前缀作性能对照。保留所有
正负结果与旧运行，不追加同键择优。数值证明缺口 `n_correct=null` 不补造。
若没有完整改善，记录当前组合的实际边界，不继续调此库/GC/整图实现。

启动前核验原 150/100 GiB 磁盘、102 GiB 空启动内存、原 72/80 GiB 服务
与 3/4 GiB 辅助约束、GPU idle、健康的既有远端。测量期间不压缩/构建/
测试/push/维护远端。旧 147 项、用户 manifest 和 EXECUTION_STATUS 零覆盖。
共享 k3s 不动，HydraServe 已有隔离适配继续复用。原远端服务不重启或停止。

PLAN SHA `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`；
V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`；
V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。

当前登记尚无 D1017 性能结果。system-profile 变更清单：只新增本文、entry.json
及原始验证记录；生产代码、观察器和库均未改。后续按 run-experiment 复用
已受限长任务入口，github-sync 在测量前备份。目标前不绘图。

## 入口验证已完成

当前 loader 的 `NativeDescriptorSelection` / `ProductionDescriptorPath` 共
9 项既有测试全部通过；实际 D747 库成功在 torch 2.13.0+cu130 / CPython 3.12
载入。对同一个 CPU Tensor 的 storage 从 64 B 增至 256 B，紧接着读取可见
新容量及当前指针/pinning，CUDA 未初始化。这只验证当前 ABI 和实时读取，
实际 GPU 九状态仍复用 D747，不声称又测了 GPU 性能。

受限验证资源峰值 616013824 B，high/max/OOM/OOM-kill/swap 均为 0；验证
任务已退出。原始来源与 SHA、资源身份保存于同目录 `validation.json`。
没有生产源修改、额外 profiling 或新资格重跑。下一步为备份后一次完整 W2；
实际生效以本次 worker 回执为准，结果未出前不作收益判断。

未完成的最终同版本 W0/W1/W2、G1/G2、baseline/dLoRA 剩余点、A1–A5、
S1–S13、正确性范围、审稿文档/图表和最终 V2 集成全部保留，不以此次恢复
启动配置代替完成系统优化。
