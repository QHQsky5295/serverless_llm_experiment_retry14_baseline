# D798：Loquetier 3B 真实输入的完整模型对照

2026-10-24 13:27 CST。一次有界数值检查完成，不是完整性能回放或全池正确性证明。
本轮不改 Loquetier 的算法、调度或模型服务代码；只为既有 runner 增加显式
模型对照入口，复用 D797 的模型专用内核。D797 的必要适配及旧 D796 失败
均保留。22项 CPU 回归通过，D798 执行后没有修改实测 runner。

## 输入、路径与判据

- 复用 D790 保存的两个真实输入：`req_00200`（760 tokens）、`req_00001`
  （477 tokens）；不生成新 prompt，不采用 D790 输出作为参考答案。
- 对应逻辑 adapter 为 `finance_lora_0025`、`finance_lora`；现有只读工件
  内容不变。两份 payload 的 SHA 相同：
  `dfd99d29ccc482c8634823bef3eb290a928e79e9de52bcf0fdabc5b581ab7bf3`。
  因此这是两个逻辑绑定、一个独立权重内容，不声称独立训练模型多样性。
- 原生 `MixedLlamaForCausalLM`、`MixedLoraModel`、SMLM 和 `ModelGeneration`
  实际运行；两请求共同 prefill，再共同 decode 两步。只读观察原生 greedy
  决策，不替换 token、不改变 logits。输出 head 实际与输入 embedding 共用
  storage；112个目标模块、rank8 的原生 LoRA 指针映射检查通过。
- HF/PEFT 参考在原生模型释放后顺序载入，FP16、Transformers4.45.2、
  torch2.1.2+cu121；以原生已选 token 作为两边共同的后续输入。
- 沿用 D790 的有界数值判据：relative L2≤0.01、cosine≥0.999、greedy
  token 相同。不是论文性能阈值，也不表示每个 logit 逐比特相同。

实际原生路径核查依据包括
[官方 ModelGeneration](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/src/loquetier_src/model_generation.py)
及[同版本 Transformers Llama](https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/models/llama/modeling_llama.py)。
没有更换后端、重新生成权重、复制工件池或进行 baseline 参数搜索。

## 六步数值结果

| 请求 | 步骤 | relative L2 | cosine | native/reference token | 通过 |
|---|---:|---:|---:|---|---|
| req_00200 | prefill | 0.004626459 | 0.999985695 | 578 / 578 | 是 |
| req_00200 | decode1 | 0.003549285 | 0.999989450 | 19114 / 19114 | 是 |
| req_00200 | decode2 | 0.004586586 | 0.999986351 | 374 / 374 | 是 |
| req_00001 | prefill | 0.003664394 | 0.999989390 | 578 / 578 | 是 |
| req_00001 | decode1 | 0.006469116 | 0.999972939 | 2326 / 2326 | 是 |
| req_00001 | decode2 | 0.005846337 | 0.999977589 | 1288 / 1288 | 是 |

最大绝对 logit 误差0.05859375；全部有限且通过预定标准。原生生成结束后
两请求和 KV 页均为空。GPU 峰值 allocated7,670,029,824 bytes；载入参考前
allocated75,628,544 bytes，不声称进程尚存时 GPU 已完全释放。

实际 service/watchdog 均返回0，服务路径移除、NVML确认上下文释放，辅助
资源域已关闭。21个保护采样中服务最大current7,452,946,432 bytes、主机
最小available107,719,073,792 bytes，high/max/OOM/kill全0，没有保护中止。
采样峰值不是精确瞬时峰值，也不将该小样本内存等同于500池容量。

## 证据身份及下一步

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d798_20261024`。
实测 runner SHA：
`5051760453bfa3a8be9578073a227a844f2da6d9927fb44a62672111f80da20d`。
输入 receipt SHA：
`c81d3376854fcc9d8f6b8ebb431908ae49df86c6a2d5d08cc3eb7f8de43c598d`。
模型专用内核 SHA：
`b1f4b70ef8d816183a9d3ca0211b79a022f05af16a9228f34346bbe4ef2e1d41`。
其余环境、配置、原生源码、输入及清理身份见 native_probe/launch JSON。

curated目录为`paper_results/ieee_tc/baseline_audit/d798_loquetier_model_output/`。
不把六个正确 token 外推为4000请求或500个ID数值正确；distinct权重限制
保留，不重开已经完成的工件填充审计。

下一复用既有四分片、顺序启动、原生批处理、真实远端和生命周期测量框架，
绑定3B的静态容量/内核/源视图。不能把7B D565容量数据改名当作3B测量；
必要的3B容量适配有界完成后执行既定完整矩阵。HydraServe3B随后推进，
不改其他项目k3s。全部主比较、PrimeLoRA优化、同版Full、消融、敏感性、
成本、审稿文档、图表及最终V2仍保留，全球最优未获证明。
