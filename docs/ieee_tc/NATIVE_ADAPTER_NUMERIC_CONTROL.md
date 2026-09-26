# 7B 同提示原生数值对照（2026-09-26）

## 问题与结论

在保持已有权重和原始提示不变时，检查“生成文本相同”是否足以判断 LoRA
没有数值作用。五次诊断全部生成原定 217 个 token，文字/token ID 完全相同；
但首 token 的原生概率不完全相同。因此**不能只凭生成文字判定 adapter 身份**。

这也还不是完整数值正确性通过：同一非零 adapter 的 A/A 重复自身有概率差异，
只有一个提示、固定执行顺序，不能事后选择容差，把差异都归因于 LoRA 内容。
Full 正确性、500-ID 覆盖及 3B 非零控制仍未完成，不从本诊断推出系统性能优势。

## 固定的输入与对照

沿用 7B seed42 主 trace 的 `req_00003`，canonical prompt 和原生 input IDs 一致。
使用 vLLM 0.30.0 的 stock AsyncLLM；不经过 Prime 的 demand-load/reference 路径。
原生 scheduler 只用于观察。FP16、原配置和编译缓存复用，没有新训练、下载、
复制权重池或生成 trace。

按照原前 100 请求中的出现次序和已完成内容审计选择，不按输出挑选：

| 顺序 | 条件 | 内容 | 原生输出/目标 |
|---:|---|---|---:|
| 1 | A | finance_lora，rank8，A/B 含非零值 | 217/217 |
| 2 | Z | writing_lora_0011，rank8，A/B 全零 | 217/217 |
| 3 | B | medical_lora，rank8，A/B 含非零值，SHA 与 A 不同 | 217/217 |
| 4 | 无 LoRA | 显式基座对照，不是失败 fallback | 217/217 |
| 5 | A 再次运行 | 与条件 1 同一权重、ID、prompt | 217/217 |

全部输出 token SHA：
`046f60aa73f688614eec503afad97295bac13a72397309ead4081c1c57de7b3b`。
运行前重新核对三个 adapter 的权重/config SHA 与内容审计一致。

## 数值结果表

使用原生 `SamplingParams(logprobs=20)`。比较**第一个输出位置**的相同 token ID，
避免后续生成前缀变化混入概率比较。五组均有相同的 20 个 top-token ID。

| 比较 | 共同 token 数 | 最大绝对 log-probability 差 |
|---|---:|---:|
| A 首次 / A 重复 | 20 | 0.00636733 |
| A / Z | 20 | 0.02607775 |
| A / B | 20 | 0.01218653 |
| Z / 无 LoRA | 20 | 0 |
| A / 无 LoRA | 20 | 0.02607775 |

原始概率保留，不是文本重分词，不是完整词表 logits，也不是独立 PEFT 参考。
可以观察到非零工件条件下的概率变化，不能将一次 A/A 差值作为已经标定的数值
误差上界；这不是五次独立重复，不能计算显著性。日志也存在首次形状 JIT，
因此不把诊断耗时当 warm latency，亦不据此猜测 A/A 差异的具体 kernel 根因。

采用正确性状态表而非性能图，遵循计划 §11 和 academic-plotting 的问题—数据
对应原则。没有统计排名，也不视觉放大微小差异。

## 接口与依据

复用已有 `scripts/ieee_tc_preflight.py backend-model-check`，仅新增 opt-in
`--qualification-mode native_numeric_reference --artifact-audit <completed-json>`。
仍用原受限启动入口，控制组未加入任何正式 runner。没有修改生产采样、路由、
缓存或 admission 策略。新测试拒绝把零工件作为非零参考、混用 rank、改变 prompt
或接受非有限概率；还验证相同 argmax 不掩盖不同概率。

官方 0.30.0 说明 `logprobs` 返回候选和采样 token 的对数概率：
[SamplingParams](https://docs.vllm.ai/en/v0.30.0/api/vllm/sampling_params/)。
本机同版本 `sampling_params.py` 和 `outputs.py` 的返回结构已核对。
本实验没有启用 batch-invariance 来事后改变共同后端，也没有替换生成方法。

## 资源、来源与下一主线

服务使用共同 72/80 GiB high/max、2 GiB swap 上限及 CPU 集；独立 watchdog
记录 71 次采样，服务峰值 5,059,387,392 bytes，最低主机可用内存
106,164,277,248 bytes，high/max/OOM/OOM-kill 均零。三个 adapter 移除成功，
native scheduler 无残留请求/iteration/deferred-free，GPU contexts 清空，服务域移除。

- 原 run：`llama2_7b_nativenumeric5_attempt1`。
- 原始 SHA：`05c23143f1dd313e0537324259bc8c17463dd8cb52ea3f8fe1b821885724e83f`。
- 汇总：`paper_results/ieee_tc/p2_backend/20260926_7b_nativenumeric5.{csv,json}`。
- JSON 保存 trace、工件、checker、原结果/启动/日志 SHA 与资源证据。
- 本次只有正确性/观测补充，不是 M1/M2、消融或层级 motivation 实验。

停止重复这类同提示输出测试，回到 P1/Full 的前决策 source/cost、原子资源
reservation/admission 和物理 GPU 生命周期接入。独立数值参考作为尚未通过的
资格项保留，不能悄悄取消；3B 新增非零正确性工件仍需用户确认范围变更。
实际 baseline 资格仍按 Serverless 优先，远端磁盘门槛仍未解除。

## D63：独立 HF/PEFT 参考（执行前固定）

上一项只有stock vLLM自身的A/Z/B/base/A观察，没有独立数值参考。现在复用
那份原生记录和同一个原始`req_00003`，补独立模型实现，不重复生成217token
文本或筛选提示。原始native结果SHA保持
`05c23143f1dd313e0537324259bc8c17463dd8cb52ea3f8fe1b821885724e83f`。

参考使用已安装旧环境`LLM_vllm0102`的Transformers/PEFT；实际包版本入结果。
新vLLM环境不安装PEFT、不改依赖。沿用既有受限启动和独立watchdog；单GPU0，
72/80GiB high/max、2GiB swap及共同CPU集；离线模式、本地现有safetensors，
不复制/合并/训练/下载权重，不生成新trace。加载当前基座前保存选定safetensors
分片、配置与tokenizer的SHA/文件身份，执行后验证没有变化。

参考合同：

- 从原trace恢复原prompt，复用现有renderer，并要求prompt和带特殊token的
  input IDs SHA均与已有native记录一致。
- 参考采用FP16、eager attention、eval/inference模式，关闭PEFT默认的adapter
  FP32自动提升；不是改变Prime生产后端。所有载入adapter tensor的键、dtype、
  数值必须逐项与原safetensors完全相同。
- A/Z/B/base/A顺序与原记录一致；base是明确禁用adapter的独立对照，不是加载
  失败fallback。仅计算共同prompt后第一个输出位置，保存完整词表logprob与
  native原来20个token的对应值，不生成一份新的执行负载。
- 正确和错误对照一律使用事前固定`atol=1e-2,rtol=1e-2`；来源是
  [vLLM0.30采样概率与HF比较测试](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/tests/v1/sample/test_logprobs.py)。
  这不是LoRA特有误差保证，不能凭该容差下“close”就宣布身份已识别。
- 保存全对照匹配矩阵，以及HF自身A/base、B/base、Z/base、A/A的完整词表
  最大绝对差和L2差。错误adapter/base同样close时明确记为不可区分，不调小/
  放大容差追求通过，不以相同argmax替代数值正确性。

参考PEFT方法及dtype语义已核对本机0.18.1源码，并参考
[官方PeftModel API](https://huggingface.co/docs/peft/v0.18.0/package_reference/peft_model)。
原生历史记录没有基座权重内容SHA，所以本次当前基座哈希不能反填为历史锁定。
需要最终严格对齐时须将新native观察绑定到该当前内容身份，不能把历史路径相同
说成历史哈希相同。单prompt/首位置也不替代500-ID、并发/切换/完整生成资格。

复用`ieee_tc_preflight.py backend-peft-reference`，结果用正确性表，不画性能
排名。`pass`只表示参考观察完整完成；`matched_reference_consistency_pass`与
完整语义资格分开，后者保持false，直到其要求被独立满足。最多两轮有新证据
的最小检查；不陷入重复同提示试验。执行、收尾和结果回执随后记录。
