# D489–D491：原生模型加载与合并共享 LoRA 的实际 GPU 检验

2026-10-13。必要正确性适配，不是 baseline 性能优化，也不是完整 HydraServe
性能结果。保留专用 vLLM 0.4.2、原生模型与 LoRA 加载、LRU、激活、槽位、
流水线及 consolidation 策略；不接触其他项目的 k3s。

## 实际接入与保真依据

`runtime/lora_worker_binding.py` 对官方 ModelRunner 整文件 SHA 固定，生成两个
加载边界的小型 overlay：初始模型注册 D479 全局层号 manager，目标模型注册到
同一个 manager。初始和目标的 LoRA 分配均在 CudaMemoryProfiler 中；实际 CUDA
同步完成后加载调用才返回。原生 GPUExecutor 在此之后才设置 dest_inited。
异常直接向上传递，不发布一个未完成 LoRA 绑定的目标。

分段坐标来自固定 SHA 的 native partition manifest，不假设均匀切分；逐次核查
rank、pp_size、总层数、实际模型层数，且同一 worker 的 manifest 不可变。
实际 native models/worker_manager/layers 来源 SHA 也在绑定时核验。
逐字节逆变换恢复完整原 ModelRunner，证明其他执行、采样、KV 等代码未被改变。
官方干净 checkout 与已安装的原生库未修改，后续实际服务须装载此 overlay。

参考[HydraServe 官方源码与部署说明](https://github.com/LLMServe/hydraserve)，
实际固定 commit 为 `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`。
源码中初始模型有 LoRA manager，load_dest_model 没有对应注册；合并复用物理层
对象，因此不能简单为目标另建一个会覆盖共享层 batch mapping 的 manager。

## D491 实际结果

在一个受限 GPU 容器中，复用既有 Llama-2-7B 权重和 finance/medical 两个不同、
非零 LoRA，完整文件 SHA 与 D488 相同。没有新权重、工件池或磁盘分段副本。

- 原样 native get_layer_distribution 得到 pp4=[8,16,24,32]、pp2=[16,32]。
- 实际构建 pp4 rank3 的八层模型，再构建 pp2 rank1 的十六层目标。
  这是单 worker 受控分段视图测试，**不是四节点/双节点流水线实验**。
- 原生 DefaultModelLoader、LlamaForCausalLM、check_duplicate、ModelRunner 两个
  加载函数和 LRUCacheWorkerLoRAManager 均实际执行。只把权重 iterator 换为
  从已有只读 safetensors 分片读取并按原生 transform 重命名；没有 fastload。
- 初始加载74个原始 tensor，目标只加载新增72个。原八个 decoder 物理对象、
  lm_head 和已绑定 logits processor 共享；原 GPU slot ID、mapping 指针保持。
- 两个 adapter 在目标注册前已经 active。注册后再次走原生 set_active_loras，
  已 active ID 的提前返回没有造成新增层缺权重。
- 全局层16–31的32个 packed QKV/o 模块（64个 q/k/v/o 位置）通过独立 CPU
  float32 `x Aᵀ Bᵀ α/r` 比较；混合两个 adapter 和一行无 adapter。
  最大相对 L2 误差0.0002201656752731651（约0.0220%），最大绝对误差
  1.9078142940998077e-6。既有 .005 相对误差门槛及 FP16 容差未放宽。
- 实际原生线性 forward 也与 base＋独立 LoRA 更新相符，无 adapter 行更新为0。
- torch GPU peak allocated=6,790,957,056 B，reserved=6,807,355,392 B；这些是
  张量分配统计，不冒充全部 GPU 占用或生命周期成本。

已通过13项CPU测试，包括D479十项和三个新增加载边界/manifest校验测试。
提交前另跑同一测试及既有五项层坐标测试，结果见备份回执。

## 两次失败均保留

1. D489 直接导入 Llama 类，触发原生 loader/llava/llama 循环导入，发生在 CUDA
   初始化前。D490改为先导入公共 model_loader，不修改任何 native 源码。
2. D490 的受控 get_model fixture 省略了原生加载器最后的 model.to(device)。
   HydraServe 的 RowParallel weight_loader 会直接设置 param.data，导致部分
   权重保留在CPU，实际kernel明确报错“w must be a CUDA tensor”。不是 OOM，
   也不是正式服务的性能失败。D491直接调用完整原生 DefaultModelLoader，
   仅提供只读权重 iterator，避免自行重写加载语义；其余配置/容差不变。

旧源文件、错误、实际退出和释放回执均封存，未原地修正旧失败。

## 资源与证据边界

三次实际容器均在核验真实PID/cgroup/affinity之后才允许CUDA；父high10/max12GiB、
子max10GiB、swap0、CPU2,3,26,27。这是功能检查的受限包络，**不用于主性能比较**。
独立于容器的host guard核查16GiB内存/100GiB磁盘底线、900秒总限；只清理本轮
确切容器和已记录出生身份的pause。实际GPU释放、递归cgroup为空后关闭。
各次资源峰值/内存事件以curated汇总及原始closure为准，不继承旧测试的数值。

本结果证明该分段视图上的实际层号、共享 owner、active槽填充及线性LoRA数值；
**不证明完整 attention/KV、跨worker流水线、KV迁移、并发合并或端到端token正确**。
没有TTFT/TPOT/SLO或GPU-s性能结论，也不把可导入/可绑定写成HydraServe复现完成。

下一步：把D480平台接口接到D481已验证的容器/私有IPC/地址组件，装载当前overlay；
接通原生fastload与backbone provider、上层adapter身份/固定输出/native token计量，
然后100请求→500-ID→1000→4000。不能重复已经通过的CUDA/分段绑定小测代替主线。
HydraServe实际结果后才做完整可用基线比较及PrimeLoRA优化；W1/W2/G2、3B、B6、
A1–A5、S1–S13、成本与文档/最终图表范围均不减少。

原始目录：实验仓库 `results/ieee_tc/hydraserve_qualification/d489_20261013/`、
`d490_20261013/`、`d491_20261013/`。小型成功/失败证据统一在本仓库
`paper_results/ieee_tc/hydraserve/d491/`，不包含模型、凭据或大原始数据。
