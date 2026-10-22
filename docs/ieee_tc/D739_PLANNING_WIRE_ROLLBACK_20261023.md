# D739：撤回 D737，恢复已测父版

## 决定与研究含义

D738 完整 3B W0 已完成并封存，不接受 D737 的规划消息延迟解析候选。
该决定来自完整回放，而不是仅凭组件性能或测试成功判断系统收益。
具体结果与限制见 `D738_3B_PLANNING_WIRE_FULL_W0_20261023.md`。
本次只恢复生产实现，不产生新性能点，也不声称解决了 3B 的 SLO 差距。

精确恢复父版 `33b9e03cf7031bc9fb2c9518a1269746c6664beb` 的两个文件：

- `faaslora/preloading/planning_cpu.py`
- `scripts/run_all_experiments.py`

移出主动生产与测试发现范围的 `planning_wire.py` 和专用测试，按 D737
提交 `0af3cd461c58f2a2cd2fb7515e1b8e76fb5d1c11` 的字节完整保存在
`paper_results/ieee_tc/p2_backend/d739_planning_wire_rollback/withdrawn/`。
既有 Git 历史、D737 资格、D738 完整负结果及原始日志不修改、不删除。

没有改变 IEEE 九公式、路由、驻留、准入与物理状态语义，未增加 fallback、
阈值、预测器或额外运行条件。D735 的验证后发布优化仍保留；本次并非撤回
全部规划实现。仅恢复父版无需重跑 4,000 请求，不重复 D738 同键。

## 针对性核验

复用现有 `planning_cpu`、`rpc_decode`、`planning_snapshot` 三组 unittest：
41 项通过，运行 65.558 秒。两个恢复文件与父提交逐字节相同；两个归档与
被撤回提交逐字节相同。检查的 145 项旧投稿保护文件未变。
受限 CPU 检查峰值 1,755,422,720 B，high/max/OOM/swap 均为零；没有 GPU 回放。
测试使用既有 CPU 资格环境，不安装或修改后端环境。测试结束后的资源身份与
释放回执在同目录证据中保存，不能以测试结束文字代替实际释放。

一次准备脚本的 JavaScript 字符串方法拼写错误发生在生成资格脚本前，未写入
文件、未启动测试或实验；修正后资格只执行一次。保留此执行说明，不作为
系统实验失败或额外性能重复。

## 下一步与未完成范围

D737/planning-wire 列为无新前提不得重复的方向。下一先检查剩余执行计划
物化/复制是否具有独立观测证据及安全的所有权边界；这仅是待核查问题，
不是已经选定的优化，也不据栈样本比例推导可实现 SLO 收益。不得复做已关闭
的 HOST 元数据、identity-only、重复资格或同前缀 profiling。

3B 仍未达到共同 95% 联合要求，G1/G2 未关闭。其余 3B W1/W2、基线与
dLoRA 有限比较、7B 最终同版本、A1–A5、S1–S13、CE/Pareto、两轮审稿
证据文档和最终图表全部保留。共享 k3s 不操作，HydraServe 已有隔离结果复用。

## 可回退备份

使用 github-sync 的明确文件清单、测试、保护 SHA、敏感信息和 diff 检查。
由于 V2 远端存在分歧，仅推送新的 `backup/d739-planning-wire-rollback-20261023`
分支并核对远端 SHA，不 force、不宣称 V2 已同步。实际提交/远端回执保存于
原始 D739 目录；用户修改的 `lora_manifest_1000.json` 永不 stage。
