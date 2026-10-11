# S-LoRA：下一入口的只读复用核查

2026-10-11，D423 vLLM 完整 W0 正在运行期间进行的轻量只读核查。
没有启动 S-LoRA、修改运行代码、导入 GPU 框架、运行测试或批量哈希。
这不是完整资格审计，也不是性能比较；D423 终态/核验仍是当前第一主线。

## 已定位的原始记录

以下根路径均相对于 `/home/qhq/serverless_llm_baselines/results/paper_experiments/`。
只读取既有 JSON 的完成、生成、身份与计量字段，没有重算性能排名。

| 记录 | 请求 / reported success | 实际 adapter ID | 输出计数来源 | 固定长度匹配 |
|---|---:|---:|---|---:|
| 2026-04-24 主表轮次 |4000 / 4000|132|local_generated_text|未记录|
| 2026-05-13 真实远端轮次 |4000 / 4000|132|local_generated_text|未记录|
| 2026-07-18 C5 初次 smoke |100 / 100|29|slora_native_sse_token_id|85 / 100|
| 2026-07-18 FIFO 定向检查 |32 / 32|13|slora_native_sse_token_id|32 / 32|
| 2026-07-22 14:48 smoke |100 / 100|29|slora_native_sse_token_id|100 / 100|
| 2026-07-22 15:41 smoke |100 / 100|29|slora_native_sse_token_id|100 / 100|

六份记录的success字段均已逐行计数；reported success不自动等于冻结V1的
正确完成，不能只凭HTTP200或旧success标志晋升为新合同主结果。

精确来源：

1. `03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/replay/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_slora_dp4_tp1_replay.json`
2. `12_remote_fair_main_real_remote_v1/20260513_012813_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1/raw/replay/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_slora_dp4_tp1_replay.json`
3. `18_eurosys27_v2_c5_smoke/20260718T090511Z_codex_eurosys27v2_c5_smoke_llama2_7b_seed42_r100_20260718T090511Z_codex/raw/replay/eurosys27v2_c5_smoke_llama2_7b_seed42_r100_20260718T090511Z_codex_slora_dp4_tp1_fixedlen_greedy_v1_replay.json`
4. `18_eurosys27_v2_c5_smoke_targeted/20260718_094134_eurosys27v2_c5_fifo_target_llama2_7b_seed42_r32_a32_20260718T094134Z/raw/replay/eurosys27v2_c5_fifo_target_llama2_7b_seed42_r32_a32_20260718T094134Z_slora_dp1_tp1_fixedlen_greedy_v1_replay.json`
5. `19_eurosys27_v2_c5_smoke_official/20260722_1448_eurosys27v2_c5_smoke_llama2_7b_seed42_r100_a500/raw/replay/eurosys27v2_c5_smoke_llama2_7b_seed42_r100_a500_20260722T1448Z_slora_dp4_tp1_fixedlen_greedy_v1_replay.json`
6. `19_eurosys27_v2_c5_smoke_official/20260722_1541_eurosys27v2_c5_smoke_llama2_7b_seed42_r100_a500/raw/replay/eurosys27v2_c5_smoke_llama2_7b_seed42_r100_a500_20260722t1541z_slora_dp4_tp1_fixedlen_greedy_v1_replay.json`

后五份逐行检查：server_metrics均为空，没有server first/last token时刻对或
physical_owner_id字段；固定长度四份均有每请求输出token序列hash。
这是所查JSON的字段事实，不证明同目录其他原始日志不存在额外证据。
未在业务期间对这些文件做SHA封存；正式复用前仍须核对源版本和完整合同。

## 当前入口、兼容补丁与计量边界

- 项目入口 `S-LoRA_project/scripts/run_slora_fair_experiment.sh` 只委托
  `scripts/run_slora_fair_experiment.sh`，不是另一套实现。
- 环境 `slora_official_cu118`、native `/generate_stream`、adapter到lora_dir的
  精确映射已有复现资料。不能使用没有同等LoRA绑定证据的chat接口替代。
- 当前嵌套源码HEAD为 `c1ddf488781ea7f551cd0bb07bfd097124c93411`，有10个已存在
  tracked兼容修改。`docs/UPSTREAM_REPO_STATE.md` 和
  `docs/EUROSYS27_V2_FORMAL_CONFIG_GATE.md` 记录完整补丁与FIFO修正。
  本次只读查看HEAD/status及diff统计；尚未重新验证补丁压缩包与当前字节一致，
  不重置、不覆盖这些修改。
- 当前wrapper第397–424行在启动前materialize完整subset；第560–638行逐副本
  启动并等待ready；第665行以后才执行回放。当前入口不能直接满足固定提前60秒
  通知且独立于ready的open-loop合同。全静态池启动准备可以是合法的S-LoRA行为，
  但须放在共同通知后的真实时间线内，不能借ready后移动业务起点获得免费准备。
- 旧fleet摘要将各副本startup取max；这些逻辑时长不是本次真实物理GPU租约积分。
  不能从该字段补造allocation/release，CPU-only staging也不能虚构持卡时间。
- 当前replay客户端约1380行以首个非空payload计API TTFT，约1650行在缺原生
  first/last时采用API完成减TTFT计算TPOT。整数token计数与原生token事件计时是
  两个不同条件。已有FIFO修正保留事件，并不自动提供原生时刻。
- 旧wrapper包含按端口清理已有监听者、删除指定缓存目录/旧日志的行为。
  不能直接执行该旧入口来启动新合同实验；后续须复用现有受控启动/所有权校验/
  独立回放/物理租约机制，不能杀未知进程、覆盖旧结果。
- 用户修改的 `scripts/replay_openai_trace.py` 必须保留；上述读取不是修改许可。

## 下一步的范围

1. 先完成D423的实际释放、原始证据核验、封存和同合同比较。
2. 再对S-LoRA上述已有环境/补丁/固定长度/FIFO证据做按字段的R0–R3复用判断，
   查看对应服务原始日志、输入与初态，不能把“名义500 adapters”当实际500-ID执行。
3. 已通过且合同未变的部分优先复用；缺失的原生计时、pre-ready独立回放、完整
   物理租约和真实远端准备边界才是需要补齐的路径。此文不决定全部重跑，亦不
   宣称任何旧记录已成为新的主结果。
4. 任何适配先核查官方固定版本源码与历史补丁，保留S-LoRA调度、paging、batching
   和模型执行语义；不移植PrimeLoRA策略，不重新生成权重/负载。

本核查使用analyze-results的“观察—解释—边界—下一步”方式。
遵循单轮V2，不生成CI/重复均值/性能图；本页全部数值是合同覆盖计数，而非性能排名。

## 原生事件边界的补充只读核查（18:08）

已联网查阅官方固定 commit 的
[model_rpc.py](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/model_rpc.py)，
并读取当前本地 `model_rpc.py`、router、detokenizer、HTTP manager 和 SSE 入口。
这是下一阶段的计量设计依据，没有安装观测钩子、改源码或运行 GPU。

- 官方 `forward` 在采样后将 token ID/logprob 搬到 CPU，再构造逐请求输出；
  原 metadata 只有 ID/logprob，没有原生 token 事件时刻。
- 本地 router 将此 metadata 经 detokenizer 传递给 HTTP manager；现有 FIFO
  保留每次更新，但 `generate_stream` 重新构造 SSE，只输出选定字段。
  因而即使未来在上游增加时刻，也须验证端到端保留，不能假定 SSE 已含该信息。
- 候选观测边界是原生采样结果可用之后、文本解码/HTTP 排队之前；还须绑定
  实际请求、batch 的 adapter 路径和原生进程。返回客户端的到达时刻不能替代它。
  该边界尚未经过测试或与其他系统的事件语义核验，不据此宣布计量已修复。
- 后续只补观测与共同启动/资源归属，不改变原生 paging、batching、调度、
  adapter 选择或采样逻辑。需要独立 CPU 合同测试和最小路径验证，现阶段均未执行。

官方仓库说明其将全部 adapters 放在主存、按运行请求装入 GPU，支持保留其
合法全池准备行为；这不授权把准备挪到共同部署通知之前。
依据：[官方项目说明](https://github.com/S-LoRA/S-LoRA)。
固定 commit 的 API 文件本次网页读取失败；对 API 的上述判断来自本地源码，
尚不声称其已与官方逐字核验。仍优先等待 D423 完整结束，不启动 S-LoRA。
