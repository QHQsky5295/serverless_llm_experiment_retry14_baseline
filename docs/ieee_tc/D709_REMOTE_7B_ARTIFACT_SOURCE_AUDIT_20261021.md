# D709：远端 7B 工件来源与数值状态审计

日期：2026-10-21  
类型：只读来源审计；没有修改本地或远端工件，没有启动推理服务，没有重放请求。

## 结论

远端刀片机的 Llama-2-7B 冻结池包含 500 个
`adapter_model.safetensors` 文件。逐文件解析 safetensors 头部并读取全部
tensor data range 后得到：498 个文件的 payload 全零，2 个文件含非零 payload。
全部文件均为 FP16，未发现坏头部。总 payload 为 10,905,190,400 B，其中非零
字节为 33,428,182 B。

含非零 payload 的两个 adapter 是 `finance_lora` 和 `medical_lora`。它们的
远端完整文件 SHA 与本地冻结副本一致：

| adapter | 远端与本地共同 SHA-256 |
|---|---|
| `finance_lora` | `971901c3140f83f57cbb19fcc86d33b4d010f6d8fc15fbec621f0347bc31a258` |
| `medical_lora` | `d2b1dcfe42193917d0b2e0f728e735f24ad119dd2c77cb9b20c983c7f688fa5c` |

这与 2026-09-26 的本地内容审计（498/500 全零、2/500 非零）一致。当前没有
证据表明远端传输或远端服务在发布后把 7B 工件清零。

## 与 3B 审计的关系

3B 远端池是 500/500 全零（D708）。7B 则是 498/500 全零、2/500 非零；两者
不能用同一数值资格结论概括。`adapter_data.bin` 或其他名义大小填充文件也不能
作为 LoRA 参数的替代物。

## 资格边界

本审计确认了远端文件的数值状态和两个非零文件的来源一致性，但没有建立完整
的独立推理 oracle。因此它支持以下判断：

* 文件交付、SHA、tensor 形状、缓存槽位和生命周期路径的 7B 结构性实验可以
  继续按原合同解释；
* 对 `finance_lora`、`medical_lora` 可以进一步做逐请求的独立 base/reference
  输出核对；
* 其余 498 个零 adapter 不能证明非零 LoRA 语义，不能仅凭请求完成、不同 ID
  或相同输出宣称完整数值正确；
* 在独立参考核对完成前，7B 的 `n_correct` 仍保持未知，不能把本审计单独升级
  为正式 G1/G2 排名资格。

全零部分对缓存、路由、扩缩容和生命周期的相对结构观察仍可在共同冻结池下使用，
但绝对远端网络成本仍受零数据的异常压缩影响。系统之间的 miss 次数和获取次数
可以不同，因此不能把该压缩影响简单视为对所有系统完全相同；论文需将这类结果
限定为冻结工件池上的结构性观测。

## 可复核输入与方法

远端池目录：

`/home/lab14/primelora_remote_artifacts/llama2_7b_a500_v2_publicmix`

远端连接端点：`lab14@192.168.4.174:8122`。审计通过标准库解析 safetensors
头部的 `data_offsets`，以受限块读取覆盖每个 tensor payload；过程只读，不依赖
服务输出，也没有将结果写回远端。远端和本地只对两个非零文件做完整 SHA 核对，
其余文件的全量非零统计已在同一远端扫描中完成。

机器可读回执：

`paper_results/ieee_tc/p2_backend/d709_remote_7b_artifact_source_audit/audit.json`

