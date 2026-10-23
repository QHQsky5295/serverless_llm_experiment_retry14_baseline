# D774：dLoRA原生生成计时，不改变迁移/调度

## 问题与有界改动

S-LoRA 3B W0/W1/W2已完成，下一主线为既定dLoRA 3B W1/W2。
旧dLoRA回放只有最终返回，不能用完成态时刻冒充首token，也不能由
`wait_time_per_output_token`恢复TPOT。完整历史复用分析见
[dLoRA复用说明](DLORA_3B_REUSE_REVIEW_20261023.md)。本轮是必要计量适配，
不是性能参数搜索，不重新运行已完成S-LoRA点，也没有修改PrimeLoRA算法。

核对固定官方commit `75f1c439446fe194b1df8a24982ef9067841fab5`：
[RequestTracker](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/async_llm_engine.py)
保留终态才送入AsyncStream的行为；
[LLMEngine](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/llm_engine.py)
在原生worker结果返回后更新序列、再作文本解码。
复用本机既有GQA/PEFT兼容源码和DP2/TP1路径，不覆盖vendor checkout。

本轮仅扩展BASE现有`ieee_tc_serverless_measurement.py`和
`prepare_ieee_tc_serverless_stack.py`，增加dLoRA被动观察及独立source view。
source view改变LLMEngine三处观察点、RequestOutput快照和最终API元数据。
原生EngineManager、AsyncLLMEngine、scheduler、Sequence及LoRA计算文件
保持原样；相关8份源码均按当前历史兼容树SHA核对，不声称它们全为未补丁官方版。

## 时刻定义与迁移连续性

| 时刻/信息 | 原生记录位置 | 不代表什么 |
|---|---|---|
| queued | 新SequenceGroup进入原生scheduler前 | 不用旧wall-clock arrival替代 |
| scheduled | 原生schedule已选定该group之后 | 不改变选择，不造GPU kernel起点 |
| first/last | 已有worker输出返回、scheduler更新和文本解码之前的批次时间 | 不是HTTP chunk或GPU kernel时间 |
| IDs、model_id、prompt | 原生SequenceGroup及RequestOutput | 不单独证明PEFT数值正确 |
| engine_path | 每次实际schedule观察到的engine切换 | 不替代完整KV/权重传输回执 |

时间记录放在现有SequenceGroup上，原生迁移已经deepcopy它。相比此前说明中
“可以放在AsyncStream”的候选方案，这一载体更接近原生生成事件，且无需
改动stream/迁移交接。不是再实现一套迁移器，也不通过关闭迁移使计量容易。
序列ID允许原生insert重新编号，请求、adapter model_id、prompt、输出前缀和
共同Linux monotonic时钟必须连续；迁移间隔保留在首末token差中。

最终响应携带原生快照和token IDs，不新增中间HTTP返回、不额外同步CUDA。
终态通知较晚不能推迟last-token时刻；单token的TPOT为N/A。首token是
原生可用时刻，原API终态才交付的限制仍须披露，不能称浏览器已在该时刻收到了token。
共同回放尚未接入这个adapter，因此本轮不产生性能排名或新TTFT数字。

## 已完成验证

2026-10-24，在4GiB、swap0、CPU2/3/26/27的已有辅助资源域运行：

- 12项dLoRA针对性CPU测试通过：原生RequestOutput、最终API字段、
  AsyncStream/RequestTracker终态返回、原生insert重新编号、迁移deepcopy
  后首末边界、快照不别名、错误身份/时钟/数量/前缀/生成合同被拒绝。
- 30项既有S-LoRA测量回归通过。没有改变其原生路径，不重跑GPU性能点。
- 示例合成时间线在迁移后仍保留first=12s、last=15s、3tokens，TPOT=1.5s；
  该值只是确定性单元测试断言，不是实验性能结果。
- source view独立生成、禁止覆盖；8份原生源SHA、145项投稿保护及用户
  dirty文件均核对。所有文件/日志存新D774目录。

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d774_20261024/`。
未导入完整旧dLoRA推理环境、未启动Ray/GPU、未操作远端或他项目k3s。
计量检查通过不代表数值正确性、受限服务启动或完整500-ID/W1/W2已通过。

## 下一步与停止边界

继续复用现有TC服务入口接入canonical prompt IDs、固定输出、静态
adapter/model_id映射、私有Ray与实际GPU allocation/release；不要直接运行
带`ray stop --force`/自动接他项目集群的旧wrapper。对这些实际改动做必要
验证，复用已通过的被动时钟测试，不再为无变化路径重复广泛资格。
保留历史DP2/TP1和周期迁移参数，不扩展7B、DP4或吞吐/迁移参数搜索。
之后完成限定W1/W2，再Loquetier/HydraServe 3B、PrimeLoRA优化及全计划。
