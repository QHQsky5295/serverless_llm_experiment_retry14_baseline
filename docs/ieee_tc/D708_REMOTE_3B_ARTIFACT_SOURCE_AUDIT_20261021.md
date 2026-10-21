# D708：远端 3B 工件来源与历史运行边界审计

日期：2026-10-21  
类型：只读来源审计；没有修改本地或远端工件，没有启动推理服务，没有重放请求。

## 结论

当前远端刀片机的 3B 冻结池中，500 个 `adapter_model.safetensors` 的 tensor
payload 全部为零。该结论来自远端目录本身，而不是由本地副本推断：逐个解析
500 个 safetensors 文件的 tensor data range，结果为 500/500 个文件的 payload
非零字节数为 0，合计 payload 字节为 5,963,776,000。

`adapter_data.bin` 不是第二套 LoRA 参数。生成脚本
`scripts/generate_lora_adapters.py` 的 `generate_adapter_synthetic` 明确以
`torch.zeros` 写入 A、B 矩阵，并在 safetensors 小于名义大小时以 `b"\\x00"`
追加填充文件。远端样本 `code_lora` 和 `code_lora_0015` 的
`adapter_data.bin` 均逐字节为零；它不能恢复或替代 LoRA 权重。

远端目录与本地冻结目录的样本文件全 SHA 一致。已核对的
`code_lora`、`code_lora_0015`、`medical_lora` 和 `writing_lora_0491` 均一致；
例如 `code_lora/adapter_model.safetensors` 的全文件 SHA 为
`dfd99d29ccc482c8634823bef3eb290a928e79e9de52bcf0fdabc5b581ab7bf3`，远端与
本地相同。远端文件时间为 2026-05-09 12:40，远端一次性归档时间为
2026-05-11 15:58。没有发现“远端有非零、传输后本地才归零”的证据。

## 为什么先前实验没有发现

先前的实验和 D78/D80 主要验证服务系统性质：请求完成、固定长度生成、adapter
ID 绑定、文件集合/大小/SHA、HTTP 覆盖、压缩归档和资源生命周期。D78 的 3B
发布记录包含 500 adapters、4,000 files、源 payload 19,482,573,296 B 和归档
SHA；D80 记录了 500 个配对传输、请求期打包为 0。它们没有读取 safetensors
tensor data range，也没有建立非零参考输出，因此不能发现全零参数。

历史执行记录在 D53--D66 已把当前 3B 池记为 500/500 all-zero，并明确指出
同一 prompt 下不同 SHA 输出相同不能区分 adapter 应用。D703 后才再次把这个
边界写入完整 3B 资格审计；D704（2026-10-20）对本地冻结池完成了 112,000
个 tensor、2,981,888,000 个元素的逐元素审计，非零元素仍为 0。也就是说，
“以前没有检查”是不准确的：早期检查先做了结构和传输完整性，数值审计较晚且
最初范围是本地冻结池；远端数值检查直到本次才补齐。

实验历史因此分为两类：

* 依赖 3B 池的服务路径、传输、缓存、生命周期、TTFT/TPOT 或 GPU-s 结果，
  可以继续作为**结构性系统负载观测**，前提是其请求和资源审计本身合格。
* 声称 3B 非零 LoRA 数值正确、adapter 语义区分、领域效果或 G1/G2 正式排名
  的结果，均没有足够证据，必须保持 `n_correct=unknown`、
  `numerical_adapter_correctness_qualified=false` 和 `ranking_eligible=false`。

这不是把历史运行全部判成“做错了”，而是把历史主张限定到它们实际测量的
范围；历史原始结果、日志和 SHA 不改写。

## 证据链

| 环节 | 当前证据 | 能证明什么 |
|---|---|---|
| 生成 | `generate_adapter_synthetic` 中 A/B 使用 `torch.zeros` | 当前池的构造路径与零权重相容 |
| 本地冻结池 | D704：500 文件、112,000 tensors、0 非零元素 | 本地文件的数值事实 |
| 远端目录 | 500 文件全量 raw payload scan，0 非零 payload 文件 | 远端文件的数值事实 |
| 远端/本地身份 | 多个代表文件全 SHA 一致 | 远端不是另一套隐藏内容 |
| 交付归档 | D78/D80 的源索引、内容清单、归档和 HTTP 覆盖 | 文件交付完整性；不含数值正确性 |
| 运行记录 | D703：4,000/4,000 终态，`n_correct=null` | 系统运行合同；不含 LoRA 数值 oracle |

## 对后续主线的处理

1. 不删除、替换或修复当前 500 池；不把 `adapter_data.bin` 当参数文件。
2. 不重跑同一零权重运行来“证明”数值正确；重复运行不会产生可区分 oracle。
3. 3B 的 G1/G2 和 3B 与 baseline 的数值胜负比较在非零参考闭合前保持未准入。
4. 7B 若使用已有非零池，沿用同样的远端目录、归档、传输和运行期数值审计，
   独立验证后才允许进入正式排名。
5. 若以后要补 3B 数值资格，只能使用有可信来源和可复核 SHA 的既有非零权重，
   并以独立 base-model/adapter 输出或中间张量参考建立正确性；不能用生成输出
   自身作为 oracle，不能改变原 500 池来提高结果。

## 复现指针

* 本地生成脚本：`scripts/generate_lora_adapters.py`，`generate_adapter_synthetic`
* 本地数值审计：`docs/ieee_tc/D704_3B_ADAPTER_NONZERO_AUDIT_20261020.md`
* 早期边界记录：`docs/ieee_tc/ARTIFACT_CONTENT_AUDIT.md`、
  `docs/ieee_tc/EXECUTION_HISTORY_D53_D66.md`
* 远端发布记录：`paper_results/ieee_tc/remote_qualification/20260927_d78_3b_publication.json`
  和 `20260927_d80_3b_coverage.json`
* 远端地址：`lab14@192.168.4.174:8122`，池目录
  `/home/lab14/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope`

