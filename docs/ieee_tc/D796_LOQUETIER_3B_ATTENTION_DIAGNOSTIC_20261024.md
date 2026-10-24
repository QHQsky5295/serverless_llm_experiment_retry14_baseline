# D796：Loquetier 3B 原生注意力兼容性数值诊断

## 结论及证据边界

2026-10-24，完成一次受限注意力单元诊断，不是性能工作负载。保留已有环境、
已安装二进制和全部模型/LoRA/trace；只扩展原有
`Loquetier_project/scripts/run_loquetier_trace_gate.py` 的显式诊断入口。

两个此前的源码疑点已有具体数值证据：

1. GQA group=3 的 prefill 出现非有限输出，而写入页式KV的数据与输入逐元素
   相同。相同流程的group=1/4符合普通RoPE参考，group=3的decode也符合。
   这不能作为Loquetier原算法的性能劣势，但说明当前3B原生内核不能直接用于
   新的正确性合格主结果。
2. 对普通RoPE合格的五个路径均不符合实际Llama-3.2配置的分频RoPE。
   这是模型语义差异，不是请求超出8192 tokens之后才发生的变化。

没有验证完整模型、tied head、LoRA输出或500-ID工作负载。没有测量本问题的
完整模型logit误差或时延影响；旧3B HTTP成功仍只代表接口/完成证据。7B
Llama-2配置及既有结果不受此诊断直接否定。

## 测量方法与结果

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d796_20261024/`。

- Native环境：`loquetier_20260520`，PyTorch 2.1.2+cu121、Transformers 4.45.2。
- Native kernel SHA：`100efcf29d3f8a53d8b2fe31b877d1589634024c742c8fff6ff30110f5a57ec9`。
- 3B config SHA：`39fb36dc5416f445ebc4e71cb71fbcf6727e80a35836d8ba1a1474c318467b7a`。
- 执行时runner SHA：`b05c48a045dbbd77a8405d7ac909ee86a084ada3090d4e809ab2fff56af56903`。
- 固定seed796的FP16单元测试Q/K/V；两序列17/257 tokens，追加一token后
  decode长度18/258。不是新生成的论文工作负载或LoRA工件。
- 实际head_dim128、KV heads8；group3是该模型，group1/4为索引对照。
- 参考：显式FP32 causal attention、split-half RoPE，以及已安装官方
  Transformers的default/llama3频率计算。输入使用同一批FP16值转FP32。
- 诊断allclose为atol=rtol=0.005，不是SLO、完整模型正确性标准或性能门槛。

| Group | 阶段 | 相对普通RoPE L2误差 | 相对Llama-3.2 RoPE L2误差 | KV内容相同 |
|---:|---|---:|---:|---|
| 1 | prefill | 0.000319646 | 0.012208463 | 是 |
| 1 | decode | 0.000198619 | 0.011007932 | 是 |
| 3 | prefill | 非有限，不能计算 | 非有限，不能计算 | 是 |
| 3 | decode | 0.000207971 | 0.012908449 | 是 |
| 4 | prefill | 0.000316330 | 0.012577480 | 是 |
| 4 | decode | 0.000207367 | 0.013215446 | 是 |

普通RoPE五个有限路径均通过上述数值容差；Llama-3.2六个路径均未通过。
两种频率最大绝对差为0.000984656740911305。GPU allocator峰值
121020416 bytes，仅为小型计算诊断，不是服务容量或GPU生命周期效率结论。
服务和watchdog退出码均0，实际GPU释放、服务路径移除、辅助资源域关闭。
数值不符合仍完整保存；程序完成不等于模型正确性通过。

## 历史与官方实现核查

先读既有`LOQUETIER_3B_COMPATIBILITY_AUDIT_20261024.md`和历史3B资格记录：
group3已经编译，但此前没有独立数值对照；不再重复单卡500池OOM或重新审计
零填充工件。D589/D560的7B SMLM、静态分片和观测路径继续复用。

本轮按research-lookup进行有界原始来源核验。Parallel CLI仍未认证，改用
可用网页工具，没有安装依赖或上传私有代码/数据。

- [Loquetier固定commit的prefill](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/kernel_src/third_party/flashinfer/include/flashinfer/prefill.cuh)：
  原有索引把8/group和16/group提前做整数除法，不能保持group3的组内头余数。
- [Loquetier原生模型路径](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/src/loquetier_src/models/mixed_llama.py)：
  推理prefill/decode只向注意力传递theta，训练rotary_emb不能证明推理已兼容。
- [Transformers v4.45.2频率定义](https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/modeling_rope_utils.py)：
  llama3按波长分频并在中间频段插值。
- [FlashInfer v0.2.0原生prefill](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.2.0/include/flashinfer/attention/prefill.cuh)：
  采用完整packed query索引再divmod得到token与组内头，为必要索引修正提供
  原始实现参考；没有因此升级整套推理系统或替换SMLM。

## 下一必要适配，非性能搜索

首选在新隔离源视图中修正packed row语义：对完整行号p使用
`token=p/group`、`head=p%group`，并使query读写、RoPE位置、causal mask、
tile起止和最后部分tile一致。不能仅解除assert或加入dispatch3就宣称支持。
RoPE按真实config补齐低/中/高频处理；prefill/decode必须一致，不能只改theta。
先验证受影响的注意力，再复用现有真实输入检查完整prefill/decode、LoRA绑定
及tied head。后者仍未完成，不能提前宣布3B可用于G1/G2主比较。

保留原生SMLM、静态分片、队列/批处理和已冻结性能配置；不替换成其他系统，
不新增参数搜索，不因为SLO低而延长基线优化。必要正确性完成后执行3B
W0/W1/W2，再HydraServe 3B。已有他项目k3s不改；HydraServe复用此前隔离部署。

随后仍是完整基线比较、PrimeLoRA优化、同版Full主表/消融、全部A1–A5与
S1–S13、成本/CE/Pareto、两轮审稿文档、图表及最终V2；未删减计划，未证明
全局最优。本轮不生成性能图、不修改论文、不新建工件或覆盖旧结果。
