# D775：dLoRA 共同输入与原生 adapter 绑定

2026-10-24。PROGRESS；不是 GPU 性能实验，不宣布 dLoRA 资格全部通过。
继续 D774 后的既定主线：dLoRA 仅 3B W1/W2 → Loquetier/HydraServe 3B →
完整对照后的 PrimeLoRA 优化。其余消融、机制、敏感性、文档和图表不删减。
最新用户对集群归属的说明已核对：现有 k3s 属于其他项目，不修改其配置、
权限、服务或工作负载；后续复用本项目已建立的隔离适配，不借用该集群。

## 1. 原因与修改边界

D773 历史复用审计已确认：旧 dLoRA 完整结果的生成合同/计量不符合本轮，
不能仅通过重算成为新的 W1/W2 性能。D774 已补被动原生事件计时；本轮只
补共同输入进入原生分配/生成路径及取消时的当前所有权，不做基线性能搜索。

重新核对固定官方版本
`75f1c439446fe194b1df8a24982ef9067841fab5`：

- [EngineManager](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/engine_manager.py)
  调用 engine.generate 时未转发 prompt token IDs；底层 AsyncLLMEngine 已支持
  这个可选参数。因此相同文本仍可能被历史 tokenizer 重新编码。
- [原生 API](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/entrypoints/api_server.py)
  的取消闭包绑定初次选择的 engine；迁移后请求流可能属于另一 engine。
  这是共同服务入口的必要所有权适配，不是改变迁移收益或迁移决策。

本地继续复用已保留的原始树、DP2/TP1 历史配置及现有 runner。无新模型、
权重、trace、全局 Ray 清理、GPU 运行、远端服务启动或新 Kubernetes 部署。

| 修改 | 保持不变的内容 |
|---|---|
| opt-in source view 给 select_engine 增加可选 prompt_token_ids 并转发 | 分配分支、评分、tie-break、放置及迁移主体；反向移除两处转发修改后与原源码完全一致 |
| 从每个原生 lora_config 读取同一零基整数映射 | 不对 adapter ID 重新排序，不改变各副本实际放置 |
| 现有 replay_vllm_engine_trace.py 内增加 NativeDloraRequestHandler | 仍调用 EngineManager.select_engine，不引入轮询/RR/PrimeLoRA routing |
| 固定 greedy、ignore_eos、目标数、canonical IDs，核对最终原生记录 | 不用文本重分词或 expected_tokens 冒充生成数量 |
| 失败/取消时在原生 select_lock 下查当前 RequestTracker owner | 原迁移、请求流和调度机制不改；只调用当前 owner 的原生 _abort，不广播或全局停止 |

source view 的旧默认行为保留；只有显式 `--canonical-input` 才加入 manager
转发。原生 API 文件保持 D774 的返回元数据适配，本轮不将它假装成已经接好
共同 HTTP 协议的完整服务；新增 handler 尚需下一步 launcher/资源计量接入。
Native AsyncLLMEngine、scheduler、SamplingParams 和 LoRA 运算源码未修改。

## 2. 已执行的验证

RAW：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d775_20261024/`

复用 D774 的受限资格脚本，只执行受影响的 dLoRA 测试模块；不重跑未变化的
30 项 S-LoRA 资格，也不启动新的诊断回放。21 项全部通过，耗时 0.061 秒，
包括原有 12 项时间线检查及新增 9 项输入/绑定/取消检查。

测试直接提取并执行原生 select_engine、AsyncLLMEngine.generate、RequestTracker
和 SamplingParams；仅将 GPU 生成替换成明确的合成输出。这是 CPU 接口验证，
不是 GPU 推理、数值正确性或真实迁移传输性能证据。

- 非字母顺序加载池及第二副本成本更低的案例，确认没有排序重绑定或 RR 替代。
- 原生 generate 接收到相同 prompt IDs、目标、模型整数索引；最终 ID/计时严格匹配。
- 非法输入、越界 token、上下文、输出数、停止条件及未知 adapter 在调度前拒绝。
- 原生返回绑定错误不被改标签为成功；重复请求和异机时钟不能二次派发。
- 迁移后取消调用当前 owner，而不是初次选择的 engine；保留取消状态。
- 原生源码树未修改；独占输出目录防止覆盖旧 source view。

资源域为 aux1，内存峰值 39,014,400 bytes；high/max/OOM/swap 均 0。
测试进程已结束，递归空域已核实并关闭。145 项旧投稿文件及三项用户 dirty
SHA 保持原样。没有性能运行期间的维护、清理或压缩。

## 3. 证据边界与下一实际动作

`lora_numerical_correctness_qualified=false` 保留。名称/整数索引、正确 token
数量及合成输出测试不能证明实际权重或 GPU LoRA 运算正确。当前 handler
也没有完整物理 worker receipt，不能进入正式 G1/G2 汇总。

接下来只补必要路径：

1. 复用现有受限 TC launcher、pre-ready ingress、实际 worker 生命周期接口，
   接项目私有 Ray；禁止原旧 wrapper 的 `RAY_ADDRESS=auto` / 全局 `ray stop`。
2. 在真实 GPU/NCCL 初始化之前绑定实际分配身份，记录退出与资源释放；
   保留 CPU-only 准备边界，不能由 ready 时间倒推整个持卡时间。
3. 使用现有 PEFT loader probe 补所选 adapter/层、声明模块和 GQA 必要检查，
   做变化路径资格；不重复已通过的未变化基础检查或搜索性能参数。
4. 完成真实远端、共同固定输入的 3B W1/W2 各一次，再进入剩余基线。

run-experiment 技能用于保护/复用既有入口；github-sync 用于显式路径检查点。
BASE 代码和 MAIN 小型证据分别备份。MAIN V2 分支此前已分叉，本次继续使用
可核验 backup ref，不 force，不把备份分支说成 V2 已同步。最终提交和远端
核验结果见 EXECUTION_STATUS；本文件不提前填写尚未发生的推送结果。
