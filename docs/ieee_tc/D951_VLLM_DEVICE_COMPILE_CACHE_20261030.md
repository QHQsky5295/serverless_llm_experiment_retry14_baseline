# D951：vLLM baseline 按物理设备隔离编译缓存

日期：2026-10-30  
来源：vLLM 7B fixed-contract smoke

## 观察到的问题

四个 TP=1 vLLM 0.10.2 副本同时启动时，共用默认的
`~/.cache/vllm/torch_compile_cache/.../rank_0_0`。首轮 smoke 中，一个进程在
TorchInductor `CompiledArtifact.save` 的临时文件重命名阶段收到
`FileNotFoundError`，其余副本随之被停止。没有发送正式 replay 请求，因此该
尝试不产生性能结果。

## 必要适配

脚本现在为每个副本设置公开的 `VLLM_CACHE_ROOT`：

```text
<device_compile_cache_root>/host-<hostname>/cuda-<physical-CUDA-mask>
```

该目录按主机和实际物理 GPU 身份稳定复用，不含 run ID；因此不会人为制造每轮
冷编译。原生 vLLM 编译、CUDA Graph、batch/token 上限、LoRA 注册、调度、模型和
生成合同均不变，也未启用 `--enforce-eager` 作为兜底。

适配依据是已审计的 ServerlessLLM D755：PyTorch 2.8 的原生 compiled-artifact
保存路径在共享目录上存在删除竞争，按物理设备隔离目录可消除该竞争。它是启动
正确性适配，不是 vLLM 性能调参。实际 cache root 写入启动日志和结果 manifest。

## 边界

该适配仅适用于每个物理 GPU 由一个 TP=1 vLLM 副本独占的当前 profile。若未来
同一物理 GPU 同时运行多个引擎，必须重新定义 cache ownership；不能复用本回执
作为通用多租户证明。首轮失败原始日志保留，下一次只验证这一适配，不把失败算作
baseline 性能数据。

