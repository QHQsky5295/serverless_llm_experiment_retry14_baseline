## 裁决

**WARN；`review_independence=same-family`，`acceptance_status=provisional`。这是有界咨询审计，不是正式或跨模型家族 PASS。**

现有证据支持：

- D1046/W0、D1047/W1、D1045/W2 各完成 4000 请求的原生生成、请求—adapter/content/slot/lease 绑定与计量合同。
- 三场景联合时延通过数分别为 **3914、3836、3833**；D1048 的资源、时延和相对差值与输入一致。
- D980/D981 是**旧 3B 配置、100 请求、29 个 adapter 的独立实现序列见证**：95 条完全一致，5 条保留不一致；不是任务数据集准确率，也不是当前完整运行的数值证明。
- 当前 3B 全零 A/B 工件，不能通过输出一致性证明非零 LoRA 效果，或区分正确零 adapter、其他零 adapter、未应用 LoRA。原生绑定和加载证据须与这项数值不可辨识性分别陈述。

**可以继续已批准的同合同机制消融。** 保留结构负载、单轮开发观测及数值证据边界即可；本审计不要求新增全量 HF/PEFT exact 重测、不重复已通过的绑定/长度资格、不重新标定 warm/预算，也不改写原始 `n_correct=null`。

## A–F 审计

以下相对路径均以 `/home/qhq/serverless_llm_experiment_retry14_baseline/` 为根。

### A. 参考来源：WARN，未发现把被测输出伪装成任务真值

D981 的 reference 由独立 HF/PEFT 执行产生，不是复制原生预测：

- `scripts/ieee_tc_preflight.py:3477` 加载基座；`:3485` 加载 PEFT；`:3494` 核对 reference 中 adapter 张量；`:3506` 选择 adapter；`:3532` 实际生成；`:3539` 提取 reference IDs；`:3544` 比较两条序列。
- D981 原始 launch 的第 5–18 行记录实际 `backend-peft-sequence-reference` 调用。
- D981 原始结果第 1517 行明确 Llama-3.2-3B；第 1618–1622 行记录 torch/transformers/peft 版本；第 1729 行为 `prompt_identity=all_match`。
- 本次独立核对了 D980 原生文件 SHA、100 个请求的 adapter/长度、捕获 token 摘要及 100 条原生绑定检查。

需要收窄的主张：

- `docs/ieee_tc/D981_3B_NUMERICAL_WITNESS_AUDIT_20261031.md:50` 的近邻解释不能凭数值本身排除缺失 adapter；`:54` 的 intended-adapter 判断应明确来自原生绑定证据，而非 95% 序列匹配。
- `paper_results/ieee_tc/p2_backend/20260926_artifact_tensor_audit.json:14`、`:23`、`:24` 记录 3B 池 500/500 全零且零更新；`docs/ieee_tc/ARTIFACT_CONTENT_AUDIT.md:30`–`:34` 已明确其辨识限制。
- 这是**独立数值推理参照**，没有任务正确答案，因此不能写成任务准确率 95%，也不能写成“造假 ground truth”。

### B. 归一化：PASS，仅限已检查的确定性计算

未发现以本系统输出的最大值、均值等作自我归一化：

- `scripts/ieee_tc/check_d299_full_context.py:1025`–`:1054` 使用共同固定长度分组阈值。
- 同文件 `:1700` 为 `joint/4000`，`:1702` 为 `U/4000`。
- `paper_results/ieee_tc/main_comparison/d1048_same_version_3b/analyze.py.snapshot:148`–`:156` 核对共同 Resident 来源后计算 `U/B`。
- `scripts/ieee_tc/check_d299_full_context.py:985`–`:987` 的相对变化分母是比较对象的实测值，绝对量同时保留。
- D648 冻结阈值见 `paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json:21`–`:37`；D657 的 W1/W2 预算见 `resident_references.json:41`–`:47`、`:73`–`:79`。

这些是系统时延、资源和约束比例，不是质量准确率归一化。

### C. 实际结果与主张值：WARN，仅两处小型说明瑕疵

确定性复核结果：

- 20 个 `source_registration.json` 登记来源 SHA 全匹配。
- `comparison.json` 的 41 个直接来源 SHA 全匹配。
- 20 行使用实际 `sealed_matrix_row` 逻辑重新投影，与存档字段一致。
- 全部相对差值重新计算一致。
- 三份请求 CSV 各有 4000 个唯一 request ID、458224 个实际输出 token；联合达标计数一致，TTFT/TPOT/E2E 均值和 Type-1 P95 重算差均为 0。
- `comparison.json`、`metrics.csv`、`source_registration.json` 与 manifest 输出 SHA 一致。

主要证据：

- `paper_results/ieee_tc/main_comparison/d1048_same_version_3b/metrics.csv:2`、`:9`、`:16`。
- 三份 `completion/full4000.json:6`–`:18`；D1045 的 GPU 时间见 `:36`，D1046/D1047 见 `:34`。
- D1048 文档第 28–47 行与这些值一致；第 57–72 行如实保留时延取舍和预算比较边界。

需要附加勘误、不改 raw：

1. D981 原始 JSON `:1513` 的模板文字误写 “7B”；实际 `:1517` backbone、`:12` config 和 `:14` adapter root 都是 3B。当前生成代码 `scripts/ieee_tc_preflight.py:3387` 仍含该硬编码文字。
2. D981 文档 `:50` 说“两条 finance 请求的 gap 为 0”，原始结果实际是 **req_00006/finance_lora** 与 **req_00078/research_lora_0008**。对应原始 JSON `:83`、`:189`、`:1213`、`:1277`。95/100、5/5 top-2 和 gap 数值本身正确。

D981 raw `pass=false`、`matched_reference_consistency_pass=false` 保留，不能改成 exact PASS。它表示该严格序列测试未全等，不自动表示服务请求失败。

### D. 实际执行的指标函数：PASS，限已追踪调用链

并非只有定义而未调用：

- D1047 collector `collect_closed_evidence.py.snapshot:37` 调用 `reevaluate_common_slo`，`:40` 调用 `diagnose_common_slo`；`:42` 明确要求结果通过且 `n_correct is None`。
- `scripts/ieee_tc/audit_d703_3b_full.py:1942` 调用 `closed_physical_integral`；`:1970` 调用 `matched_controller_rows`；`:1990` 按原生事件重建 TTFT/TPOT/E2E；`:2003` 调用 `common_slo_flags`；`:2030` 调用 `timing_stats`。
- `scripts/ieee_tc/check_d299_full_context.py:1005`–`:1007` 实际调用 `binding_checks` 和 `request_vectors`。
- `scripts/ieee_tc/plot_d290_diagnostic.py:237`–`:258` 核验 request/adapter、owner、lease、native adapter ID、slot、冻结 content、获取/释放和时钟；其 docstring 明确“不声称 kernel arithmetic proof”。
- `scripts/ieee_tc/audit_d402_serverlessllm_path.py:305`–`:309` 是实际 Type-1 分位数实现。
- D1048 analyzer `:115` 调用 `sealed_matrix_row`，`:157` 调用 `matched_relative_observation`。

D1047 记录的历史 helper SHA 是 `d7e7956…`；当前 D1048 helper SHA 是 `10645738…`。本次通过历史 commit 核对了旧 SHA，并确认 `closed_physical_integral`、`matched_controller_rows`、`common_slo_flags`、`matched_relative_observation` 四个相关函数 AST 未改变，未把后来的源码直接冒充历史执行源码。

### E. 范围／模型／版本关联：WARN

当前三场景同配置关系成立：

- D1048 analyzer `:53`–`:65` 排除三条独占 output/cache 路径后比较全部 YAML；本次再次实际比较，三份配置相等。
- 三份 audit 分别登记正确的 W0/W1/W2 replay view；同一 source trace SHA；D1048 analyzer `:127`–`:131` 要求同场景来源视图一致。
- D1048 文档 `:15`–`:19`、`:120`–`:123` 保留开发、数值未知及未完成范围。

旧见证不能无条件覆盖当前版本：

- D979 witness 配置 `configs/ieee_tc/20261031_d979_3b_native_token_witness_100.yaml:12` 为 `max_loras=8`，`:55` 为旧 `model_backend_id`。
- 当前 D1047 配置 `:14` 为 32，`:17`–`:19` 新增 active-LoRA specialization/capture/startup-memory policy，`:59` 的 `model_backend_id` 已不同。
- 因此 D981 的 95/100 应保留为**旧配置见证**；不能直接说当前 D1045–D1047 也有 95/100 或全量一致。配置变化本身亦不证明当前实现错误。

D1047 `native_descriptor.json:5`–`:8` 证明 4000 条记录具有指定 descriptor kind 和库 SHA；collector `:48`–`:51` 实际只对这些字段计数。它不等同于 4000 次独立张量值校验，更不等同于非零 LoRA 算术效果证明。

D990/D991 明确是 7B（各文档第 1 行），未用于推出 3B 数值正确性。

### F. 评价类型：PASS，必须分开分类

- **主比较：真实系统执行测量**——原生事件、绑定、时延、物理 GPU 生命周期；不使用任务数据集 GT。不能硬说成 task accuracy 或 simulation-only。
- **D981：独立实现的数值一致性参照**。若必须使用 skill 的封闭枚举，最接近 `synthetic_proxy`，但应附 subtype=`independent_numerical_inference_reference`，明确不是“被测输出自充真值”、也不是任务质量 benchmark。
- **工件审计：静态内容／零更新审计**，不构成任务质量或 adapter 语义效果评估。

## 最小后续与停止条件

本轮所需后续仅为新报告中的范围说明和上述两项勘误，原始文件不变。

机制消融可推进，使用同配置 Full 参照、冻结合同与原生绑定计量，并继续审计同后端输出变化。不能将本审计变成新的资格协议。

只有需要主张“当前版本的有界数值一致性”时，才先离线按 request、canonical/native prompt、target、adapter/content 等完整键连接现存参考与输出摘要；旧输入相同不代表执行时序和计算配置相同。若发现具体加载、绑定或计算异常，再补受影响路径的最小证据。**未知的全量算术证明本身不是要求重跑 4000 条 exact reference 的依据。**

冻结依据：V1 `:53`–`:71` 明确定义请求合同和跨后端不要求全等；V2 `:31`–`:33` 允许有来源的新派生指标但禁止原地补写未知字段。D637 `:59`–`:63`、D638 `:85`–`:94` 也明确禁止无限 exact 资格循环。

## 审计限制

没有新 GPU 执行、服务操作、文件编辑、权重哈希或大目录扫描。

未重新展开／读取：

- D1045/D1046/D1047 的大型原始请求 JSON、`main_outcome.json`、watchdog 和完整 allocation journals；本轮复用 SHA 绑定的已封存审计，并复算小型请求 CSV。
- 基座及 adapter 权重本体；复用既有内容审计及冻结索引。
- D990/D991 的 7B 原始数值结果；仅阅读其范围文档。
- `EXECUTION_STATUS.md` 除前 180 行及 D1045–D1048 定向检索之外的历史内容。其头部是旧 D998 状态；D1048 文档 `:134`–`:135` 已明确提醒不要据此重启旧工作。
- `check_d299_full_context.py`、`audit_d703_3b_full.py`、`ieee_tc_preflight.py` 中与本次实际调用链无关的函数。
- skill 的缺失 `shared-references` 文件；它们的缺失不是实验 blocker。

本次只读复算脚本首次漏复制 analyzer 对 HydraServe `source_plan` 的补充步骤而触发断言；补齐这一审计脚本步骤后 20 行全匹配。这是审计脚本首次尝试问题，不是 D1048 数据缺陷。首次输出应随 trace 保留。

## 机器可读结果

```json
{
  "audit_skill": "experiment-audit",
  "date": "2026-11-04",
  "agent_id": "/root/d1049_correctness_review",
  "reviewer_model": "inherited_available_model",
  "reviewer_family": "openai",
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "overall_verdict": "WARN",
  "integrity_status": "warn",
  "formal_acceptance": false,
  "advisory_only": true,
  "reason_codes": [
    "numerical_reference_scope_is_bounded",
    "zero_update_adapter_outputs_are_not_discriminative",
    "historical_witness_not_rebound_to_current_backend_configuration",
    "minor_reference_documentation_mismatches"
  ],
  "summary": "Current 3B comparisons establish complete native execution and binding/accounting observations with reproducible reported metrics. They do not establish task accuracy, full numerical arithmetic proof, or nonzero adapter effects. Mechanism ablations may proceed with these boundaries and unchanged raw unknowns.",
  "checks": {
    "A_reference_provenance": {
      "status": "WARN",
      "fake_task_ground_truth_found": false,
      "reference_kind": "independent_numerical_inference_reference",
      "evidence": [
        "scripts/ieee_tc_preflight.py:3477",
        "scripts/ieee_tc_preflight.py:3494",
        "scripts/ieee_tc_preflight.py:3506",
        "scripts/ieee_tc_preflight.py:3532",
        "scripts/ieee_tc_preflight.py:3544",
        "paper_results/ieee_tc/p2_backend/20260926_artifact_tensor_audit.json:23"
      ]
    },
    "B_score_normalization": {
      "status": "PASS",
      "scope": "deterministic formulas and frozen denominators inspected",
      "self_output_normalization_found": false,
      "evidence": [
        "scripts/ieee_tc/check_d299_full_context.py:985",
        "scripts/ieee_tc/check_d299_full_context.py:1025",
        "scripts/ieee_tc/check_d299_full_context.py:1700",
        "paper_results/ieee_tc/main_comparison/d1048_same_version_3b/analyze.py.snapshot:148"
      ]
    },
    "C_result_existence_and_values": {
      "status": "WARN",
      "registered_source_hashes_matched": 20,
      "direct_source_hashes_matched": 41,
      "reprojected_rows_equal": 20,
      "relative_observations_equal": true,
      "manifest_output_hashes_equal": true,
      "metric_values_mismatched": false,
      "documentation_corrections": [
        "D981 raw limitations says 7B while backbone/config/pool are 3B.",
        "D981 document says two finance requests have zero gap; raw rows are req_00006 finance_lora and req_00078 research_lora_0008."
      ]
    },
    "D_executed_metric_functions": {
      "status": "PASS",
      "scope": "recorded collector/analyzer calls and matching outputs; no GPU function rerun",
      "functions": [
        "reevaluate_common_slo",
        "closed_physical_integral",
        "matched_controller_rows",
        "binding_checks",
        "request_vectors",
        "common_slo_flags",
        "timing_stats",
        "sealed_matrix_row",
        "matched_relative_observation",
        "backend_peft_sequence_reference"
      ],
      "relevant_dead_metric_function_found": false
    },
    "E_scope_model_version": {
      "status": "WARN",
      "current_three_workload_configs_equal_except_exclusive_paths": true,
      "historical_witness_current_version_equivalence_established": false,
      "seven_b_evidence_used_as_three_b_proof": false,
      "current_native_descriptor_is_arithmetic_proof": false,
      "evidence": [
        "configs/ieee_tc/20261031_d979_3b_native_token_witness_100.yaml:12",
        "configs/ieee_tc/20261031_d979_3b_native_token_witness_100.yaml:55",
        "configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml:14",
        "configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml:17",
        "configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml:59",
        "paper_results/ieee_tc/p2_backend/d1047_same_version_w1/completion/collect_closed_evidence.py.snapshot:48"
      ]
    },
    "F_evaluation_type": {
      "status": "PASS",
      "main_comparison": "real_system_measurement_without_task_ground_truth",
      "numerical_reference_skill_enum": "synthetic_proxy",
      "numerical_reference_subtype": "independent_numerical_inference_reference",
      "numerical_reference_is_task_accuracy": false,
      "numerical_reference_is_self_copied_prediction": false,
      "artifact_audit": "static_content_and_zero_update_audit",
      "taxonomy_note": "The skill's task-evaluation enum does not precisely describe systems latency/resource measurement."
    }
  },
  "current_prime_observations": [
    {
      "run": "D1046",
      "workload": "W0",
      "requests": 4000,
      "unique_request_ids": 4000,
      "native_output_tokens": 458224,
      "joint_pass": 3914,
      "joint_fraction": 0.9785,
      "physical_gpu_seconds": 7967.349446525332,
      "max_abs_csv_metric_recompute_difference": 0,
      "raw_n_correct": null
    },
    {
      "run": "D1047",
      "workload": "W1",
      "requests": 4000,
      "unique_request_ids": 4000,
      "native_output_tokens": 458224,
      "joint_pass": 3836,
      "joint_fraction": 0.959,
      "physical_gpu_seconds": 6182.715970784891,
      "max_abs_csv_metric_recompute_difference": 0,
      "raw_n_correct": null
    },
    {
      "run": "D1045",
      "workload": "W2",
      "requests": 4000,
      "unique_request_ids": 4000,
      "native_output_tokens": 458224,
      "joint_pass": 3833,
      "joint_fraction": 0.95825,
      "physical_gpu_seconds": 8182.401820929721,
      "max_abs_csv_metric_recompute_difference": 0,
      "raw_n_correct": null
    }
  ],
  "historical_3b_witness": {
    "native_run": "D980",
    "reference_run": "D981",
    "requests": 100,
    "adapters": 29,
    "native_binding_checks_valid": 100,
    "native_capture_hash_joins_valid": 100,
    "exact_matches": 95,
    "length_mismatches": 0,
    "sequence_mismatches": 5,
    "native_token_in_reference_top2_at_first_divergence": 5,
    "raw_pass_preserved": false,
    "measurement_complete": true,
    "full_semantic_qualification": false,
    "current_full_run_n_correct_inferred": false
  },
  "claims": [
    {
      "id": "complete_native_contract_execution_3b_W0_W1_W2",
      "impact": "supported_with_sealed_audit_scope"
    },
    {
      "id": "reported_3b_latency_resource_observations",
      "impact": "supported"
    },
    {
      "id": "three_b_nonzero_adapter_numerical_effect",
      "impact": "unsupported_by_zero_update_pool"
    },
    {
      "id": "current_three_b_full_numerical_equivalence",
      "impact": "unknown"
    },
    {
      "id": "task_accuracy_95_percent",
      "impact": "not_measured"
    },
    {
      "id": "formal_G1_G2_or_global_optimum",
      "impact": "not_established"
    }
  ],
  "next_action": {
    "mechanism_ablations_can_proceed": true,
    "new_gpu_evidence_required_by_this_review": false,
    "rewrite_raw_unknowns": false,
    "mandatory_followup": "Append the numerical-scope clarification and two documentation errata; preserve raw results and exact-test failures.",
    "conditional_followup": "Only if claiming current-version numerical agreement, first join retained references/output digests under complete input/content identities and inspect the affected-version evidence.",
    "unnecessary_requalification": [
      "4000-request HF/PEFT exact equality gate",
      "repeat unchanged binding/length checks",
      "repeat existing warm or Resident calibration",
      "repeat zero-adapter negative controls seeking distinguishable outputs",
      "infer 3B qualification from 7B witnesses"
    ]
  },
  "audited_input_hashes": {
    "AGENTS.md": "sha256:4d852cae628f8749349f9180ad99f60a7beac607692903b40b839a0c2845e141",
    "/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md": "sha256:d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83",
    "docs/ieee_tc/EXECUTION_STATUS.md": "sha256:41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374",
    "docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md": "sha256:5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22",
    "docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md": "sha256:193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d",
    "docs/ieee_tc/D1048_SAME_VERSION_3B_COMPARISON_20261104.md": "sha256:249b53173a0b1537f1a840ce8996121b38c4e6d06126e3ff79cb4d7bbe63d8e8",
    "docs/ieee_tc/D981_3B_NUMERICAL_WITNESS_AUDIT_20261031.md": "sha256:8a5fa808daef1979949be5ecc30bf6b1d10ba1edb7a30c0b65b9153c3704932f",
    "docs/ieee_tc/D637_REPAIR_REUSE_AND_CORRECTNESS_SCOPE.md": "sha256:fdd72498c1355193925940e0d2d6caaef928a3b0270b5cdab013700f9beb1245",
    "docs/ieee_tc/D638_OUTPUT_BINDING_AND_REFERENCE_AUDIT.md": "sha256:49fbde33466396af2d8ce22a610da31978073adb274628784d1407d1ac852679",
    "docs/ieee_tc/D393_7B_CORRECTNESS_AND_REMOTE_METRIC_ALIGNMENT.md": "sha256:d6956812aa9cde04e55334a95f2d814454798e2a3e5e4b10eee634fa3782bf74",
    "docs/ieee_tc/D991_D990_PEFT_REFERENCE_20261031.md": "sha256:9a6cfb9340f8bc9813b9e0d5897c66ea5eb3aed483c59bf00fbbc59af4192f49",
    "docs/ieee_tc/D990_7B_BUSY_SERVICE_WITNESS_20261031.md": "sha256:e95998585913308b848480789bff1e0914c19c5620f613d88e4cb2e2bf88afde",
    "paper_results/ieee_tc/main_comparison/d1048_same_version_3b/comparison.json": "sha256:96673673663996fe42e92ae65130a8c4c9d1db2d6f2c690492e27e4e36fde2ad",
    "paper_results/ieee_tc/main_comparison/d1048_same_version_3b/metrics.csv": "sha256:a6d62c2d10c3e53ca8d83146b7ca4a9050531dc1477b29fb409a24dfed51f744",
    "paper_results/ieee_tc/main_comparison/d1048_same_version_3b/analyze.py.snapshot": "sha256:21ca31d3e334f6a2b40dd5882e8a1f96cd84139a09db1df4dbdce718599f4d7f",
    "paper_results/ieee_tc/main_comparison/d1048_same_version_3b/source_registration.json": "sha256:9e4f26e580711d204f5fbf2c31cf2b932504fde2ccc5239903a39e78c56a43fe",
    "paper_results/ieee_tc/p2_backend/d1045_event_busy_time/completion/full4000.json": "sha256:5da67242cf1ad257b6a5f4bb44aa50ef4d2398d096807b624de07d1d7f3188c2",
    "paper_results/ieee_tc/p2_backend/d1046_same_version_w0/completion/full4000.json": "sha256:9b7469ceb0aadd941284db3a66a28d62939f416a8703e45caead8912a45280db",
    "paper_results/ieee_tc/p2_backend/d1047_same_version_w1/completion/full4000.json": "sha256:9efa154f7d9322e6ccac94541a7d323064d2bd868ee8642c8245755c42fde6e1",
    "paper_results/ieee_tc/p2_backend/d1047_same_version_w1/completion/native_descriptor.json": "sha256:cbc01f33f8e29673a6641d7b4671ca3e84fcecc0109405026f47cf343996c42e",
    "paper_results/ieee_tc/p2_backend/d1047_same_version_w1/completion/collect_closed_evidence.py.snapshot": "sha256:91dd9c58812f3a1d24904c76c25415b8e774e52f1123417c7cda799c2b4493df",
    "scripts/ieee_tc/check_d299_full_context.py": "sha256:10645738c5c991c827da4c7b63895b9422a4340d92f678ff820695c0b43fefee",
    "scripts/ieee_tc/audit_d703_3b_full.py": "sha256:3abcd9d766a7dc1c780244d8eedaac8440815aa85c0e45c6d9b81ef2cf35a070",
    "scripts/ieee_tc/plot_d290_diagnostic.py": "sha256:18e96959d167c3b9b8c762a4ee4274d4443eba38610e02cd89cd0755c5664f22",
    "scripts/ieee_tc/audit_d402_serverlessllm_path.py": "sha256:8a5f1bf18944a0c168525eddbde8eaed1b61e3b05d4f6279d0f9d81e8efec473",
    "scripts/ieee_tc_preflight.py": "sha256:a15c5f21e7e30860dd5e413580726e6047245362cfcbbbc290799fff2acdcd4d",
    "docs/ieee_tc/ARTIFACT_CONTENT_AUDIT.md": "sha256:b12d369fea12fba1940ac47a4a81e9506b9d86bf79e42a94792f9b9c9e7fa221",
    "paper_results/ieee_tc/p2_backend/20260926_artifact_tensor_audit.json": "sha256:a7294504fe43d3de82c27c6000a90439ef25fe45d5c3c8ae8eaf248e28c6ece1",
    "paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json": "sha256:d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8",
    "paper_results/ieee_tc/main_comparison/d657_3b_references/resident_references.json": "sha256:f030f92f7bc426c427c6fa0ea65c8dd315a39cf52fd922973fc507117f30ea7a",
    "configs/ieee_tc/20261031_d979_3b_native_token_witness_100.yaml": "sha256:e47bc09d4c18671441d75196966d247e4a85c32fe8e4c84192da15eb52f62d1a",
    "configs/ieee_tc/20261103_d1045_3b_event_busy_time_w2.yaml": "sha256:cea1b5e669efd9142352a9a04fa921eb9a2660cd3fcfd1d65a8b3992b859a60e",
    "configs/ieee_tc/20261103_d1046_3b_event_busy_time_w0.yaml": "sha256:65c327292787353d71f1037ee779762982ec384089ebab5f0c028735d867598a",
    "configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml": "sha256:263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2",
    "/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d981_20261031/3b_peft_sequence_reference_100_from_d980/3b_peft_sequence_reference_100_from_d980.json": "sha256:11c715afbaf80eeb69c5fb318864819953e3f74a0cfeb3af03b5f3d4dcc6efad",
    "/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d981_20261031/3b_peft_sequence_reference_100_from_d980/3b_peft_sequence_reference_100_from_d980_launch.json": "sha256:2d2e24489f96b82c9a4543453edcf95b906dc8e7022b8234e1c6580729c52b88",
    "/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d980_20261031/3b_native_token_witness_100/experiment_results_full_vllm_dedicated_a500_r100_c8_tc_ieee_native_token_witness_100_d980_3b_native_token_witness_100.json": "sha256:276b6f4577c72ad3c6f96e7a5a5e85f9c4bd2a88f3a8c1f2f31b947ad20c0eac"
  },
  "audit_attempt_notes": [
    "Initial read-only row reprojection omitted HydraServe's analyzer-level source_plan supplementation; corrected diagnostic then matched all 20 rows. Preserve initial assertion in trace.",
    "No files edited, GPU execution, services, model-weight hashing, or broad filesystem scan.",
    "Executor is responsible for saving this report and tool trace."
  ]
}
```
