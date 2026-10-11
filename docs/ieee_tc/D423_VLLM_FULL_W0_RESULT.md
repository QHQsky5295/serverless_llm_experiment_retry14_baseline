# D423：vLLM 7B 原始 W0 完整单轮结果

2026-10-11。原始4000请求回放已完成并释放全部GPU。执行版本仍为
`e646f9106c8fd6fd15f54b69041bd54234fa47e1`；此文的备份commit不是执行commit。
沿用D419/D421/D422已验证的原生vLLM0.30.0路径、普通原序号round-robin、
原生LoRARequest/LRU、共同60秒准备和受控CPU/内存资源，不移植PrimeLoRA策略。

| 项目 | 实测单轮 |
|---|---:|
| offered / submitted / terminal / 原生绑定与计时合同 |4000 / 4000 / 4000 / 4000|
| 失败 / 实际token / adapter ID |0 / 458224 / 132|
| 共同D394 TTFT / TPOT / 联合达标数 |3979 / 4000 / 3979|
| 联合达标率 |99.475%|
| 真实生命周期GPU-s / 每合同有效请求 |16115.377162149176 / 4.028844290537294|
| 平均 / P95 TTFT（ms） |358.019243 / 1200.506860|
| 平均 / P95 TPOT（ms） |24.797976 / 27.177332|
| 平均E2E（ms） |3182.813643|

全部输入/生成/实际adapter与内容收据、四真实worker、物理租约及释放检查通过；
时间重算最大误差0 ms。真实远端132次交付、131480060线上字节、3300789780物化
字节、3868次缓存命中、零请求时打包。服务峰值14824255488 B，high/max/OOM/swap
为零；无保护中止。旧结果保护147项未变。

原前100条复用D385独立参考：97个完整序列匹配，3个差异保留，仅2个差异具备
可复用margin证据；覆盖4个权重类别，不声称其余3900条独立数值证明。
不修改原始n_correct和资格未知标志，不把大量零/重复权重称作500个独立任务。
身份保持development、qualification_only、nonformal；不计算CI、不重跑择优。

原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d423_20261011`。
主仓库报告：`docs/ieee_tc/D423_VLLM_FULL_W0.md`。
主仓库curated目录：`paper_results/ieee_tc/baseline_audit/d423_7b_full_w0/`。
audit SHA `7d9ac56d0db8cf3af0f87c7a5314eb272771ff6408c419bba616e8f7eceeb310`；
206文件manifest SHA `b3796a4ea857e0a5cc8bb105397bd511f6c8548065b1c20a73ca9242975be516`。
27项启动前CPU分析测试通过，日志已保留。本轮没有修改基线运行代码。

NEXT：备份与独立完整合同关联，再决定W0 Resident参考冻结/描述性比较。
不重新执行100/500/1000资格或本轮，不跳过剩余7B基线到3B；W1/W2/G2开放。
S-LoRA只读复用核查见同目录`SLORA_REUSE_RECON_20261011.md`，尚未实现或启动新实验。
