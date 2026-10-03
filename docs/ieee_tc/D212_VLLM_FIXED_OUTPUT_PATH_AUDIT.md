# D212：vLLM 固定输出契约与原生 token 计量审计

日期：2026-10-04
状态：CPU 接口审计与 7B GPU 短资格回放完成；不构成 Resident 参考或性能结果

## 1. 目的与边界

D211 的固定四运行候选在语义审计中被拒绝：虽然关闭了 handoff、层级驻留、动态转发、扩缩容和有效容量准入，但它仍以 `faaslora_full`/`ieee_confirmed` 路径选择副本，不能作为 V1 要求的普通 vLLM Resident 参考。D212 不重跑该候选，也不从其 1,624 个成功请求推导预算或性能。

本次先审计基线仓库中普通 vLLM runner 的固定输出接入，随后用既有 W0 trace 做短资格回放，目标是使后续 vLLM 与 PrimeLoRA 使用同一 `fixed_length_greedy_v1` 输入/输出合同。没有改变 vLLM 的调度、LoRA 缓存、批处理或资源配置；本次回放不使用远端制品服务。

## 2. 发现

旧 runner 即使收到 `FAIR_GENERATION_CONTRACT=fixed_length_greedy_v1`，也没有把合同传给 replay 客户端；replay 还只接受 S-LoRA SSE 的整数 `token.id`。因此 vLLM 运行无法满足 V1 §3 的“原生 token 数量”条件，继续正式比较会把 usage 或文本重分词误当作真实输出长度。

本机冻结 vLLM 环境 `/home/qhq/.venvs/primelora_vllm0300_tc_20260925` 的官方 OpenAI 协议实现支持 `return_token_ids=true`，并在 completion choice 中返回原生 `token_ids`。这提供了与 S-LoRA SSE 整数 token ID 对应、但不依赖文本重分词的计量路径。

## 3. 受限修改

基线仓库提交 `a98ee47d174e5830d1b08da56ad8b10c4e23a671`（已推送到其 `origin/main`）包含：

1. vLLM 固定合同将 `return_token_ids=true` 加入请求；replay 从 OpenAI choice 的 `token_ids` 累积原生整数 ID，并记录通用 native token audit 计数。
2. 固定合同将完成 token 来源标记为 `vllm_token_ids`，校验器要求事件数、整数 ID 数和实际完成数逐请求一致，且无非法 ID。
3. 将 `VLLM_GENERATION_CONTRACT`、两个 token 上限显式从 `run_full_fair_round.sh` 传给 vLLM runner；固定模式仍为目标长度 `min(source_expected_output_tokens, 256)`、prompt 上限 759。
4. 若响应片段没有文本但包含原生 token ID，不再被“空成功响应”误判为失败；这只修正观测合同，不改变解码。

本提交未包含基线仓库中原有的 Serverless 未提交修改。主仓库用户文件和 `configs/generated/lora_manifest_1000.json` 未被修改或暂存。

## 4. 验证证据

- `tests.test_fixed_length_generation_contract`：9/9 通过；覆盖 vLLM choice token ID、非法类型计数、固定长度来源和既有 S-LoRA 路径。
- `tests.test_fair_runner_shell_helpers` 与 `tests.test_fair_system_resolved_config`：23/23 通过。
- `py_compile`、两个 shell `bash -n` 和 `git diff --check` 通过。
- 使用既有 W0 4000 请求/500 adapter trace 做固定模式 dry-run；没有 GPU 启动，launch/replay/summary 只读规格显示合同和上限已传递。trace SHA 为 `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，adapter subset SHA 为 `aa94b21e129a5efde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`。
- 短资格回放使用同一 W0 trace/subset，7B、DP=4、TP=1、静态注册 500 adapters，最多 100 个请求；所有副本均为普通 vLLM OpenAI server，不导入 `faaslora_full` 或 `ieee_confirmed`。输出位于 baseline 仓库的 `results/ieee_tc_resident_qualification/d212_vllm_7b_seed42_q100_static_fixed_attempt3/`，replay SHA256 为 `143b376370f086ae41eacc6423161d7127774197e2a3c9a61046e3f44fa2d82f`，summary SHA256 为 `09263f38769a48ceccccb6ed0826955a2726a22cea7e5309ca53d980751a743c`。
- 资格结果：`100/100` 请求完成且成功；严格校验器确认 `fixed_length_greedy_v1` 的 target、原生 token-ID、prompt hash 和分解恒等式 gate 全部通过，completion source 为 `vllm_token_ids`；回放耗时 189.8485 s。四个 vLLM 服务在回放后正常退出，GPU 与本次服务 cgroup 均已释放。
- 本次资格运行保留了两次启动前失败尝试（旧 CLI 参数以及 vLLM JIT 的 `ninja` PATH 缺失），它们是环境/启动资格证据，不计入性能样本；对应原始日志未删除。

上述 CPU 测试和短回放是测量接口与原生 token 资格证据，不是共同 warm SLO、G1/G2、GPU 生命周期或性能排名证据。下一步普通 vLLM Resident 候选仍须独立完成：不导入 `faaslora_full`/`ieee_confirmed` 控制语义、通过静态语义审计、4000/4000 正确完成和原生 token 校验，然后才能进行 V1 要求的三次完整 Resident 参考运行。

## 5. 与论文证据的关系

D212 关闭了一个会使 Prime/vLLM 固定输出比较失效的测量缺口，但不改变 D210 PrimeLoRA Full 的结果，也不证明 PrimeLoRA 优于 vLLM。Resident 预算、共同 warm SLO、G1/G2 和主比较仍保持未完成状态；在这些条件冻结前，不生成性能图或排名表。
