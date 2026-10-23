# D769：S-LoRA 3B完整500-adapter功能覆盖

## 完整结果：覆盖通过，不是性能达标

| 项目 | 本轮观测 |
|---|---:|
| 计划/到达/提交/终态/响应 | 均500 |
| 失败 / 实际adapter IDs | 0 / 500 |
| 原生重新核验 / 实际输出tokens | 500 / 68,925 |
| 联合SLO / TTFT达标 / TPOT达标 | 0 / 109 / 0，分母均500 |
| 平均/P95 TTFT | 124,896.175453 / 329,602.098276 ms |
| 平均/P95 TPOT | 167.448494 / 227.781954 ms |
| 物理GPU时间 | 1,893.561880 GPU-s |
| 服务内存peak | 61,332,733,952 B |
| high/max/OOM/swap | 均0 |

20:38 CST自然结束。所有目标长度、共同输入、adapter绑定及四个原生worker
计量核验通过，E2E/TPOT/native重算误差均0。独立数值正确性仍未知，
`n_correct=null`，不作完整主负载排名。低SLO完整保留，不触发baseline搜索。
平均dispatch wait73,169.925866ms、service TTFT51,718.462081ms、decode
23,160.140612ms，不能把所有慢归因于网络。3B BMM适配范围继续披露。

真实远端线上1,160,474,851 bytes，物化19,482,573,296 bytes，请求打包0。
四GPU已实际释放，自有三项远端服务已停止。现有分析器完整通过，主审计SHA
`b47df192c82594712076c1da1d6c4c1f53e238df74a497d1098f3b4f34839531`。
小型证据封存在本轮目录`gpu_coverage500/`；原始记录和全部历史失败保留。
后续复用相同配置继续1000兼容与4000三场景，不重复同键。
下方启动前/启动中记录保留为历史，不表示当前仍未结束。

闭合后仅删除本轮重复下载对象：4000文件与保留原件逐字节一致，回收
19,497,287,680 B，磁盘剩余163,739,987,968 B；维护peak1,132,142,592 B，
high/max/OOM/swap均0，空域已关闭。数据回收在推理结束后，不影响测量，
原始证据与唯一工件不变。

## 20:25 CST执行更新

本轮已在独立tmux启动，初期为静态池准备/启动阶段，尚无完成结论。
MAIN `aef1bc9056102f857956816c31a7aa0b514b90b5` 已验证备份至
`faaslora_origin/backup/d768-input-qualified-d769-coverage-20261023`；
BASE `84bb7892b4aade6bed0b898fb99d1f6074073a10` 与origin/main一致。
V2远端仍未同步，未force。服务与回放隔离、独立watchdog已运行；实际链路
1000Mb/s/full。下文“尚未启动”保留为此前输入绑定时的历史状态。

## 问题与复用边界

D768已验证100请求原生输入/输出合同，实际覆盖29个adapter。本轮回答剩余
500个既有逻辑adapter是否都能通过同一原生执行路径，不能由前缀检查代替。
直接复用D768原生配置、source_view、57项CPU检查及成功GPU证据；没有新
源码、调度/缓存策略、backend升级或baseline性能调参。

这是原计划§6.7的功能覆盖，不是4000请求主性能点。当前仅完成输入绑定，
尚未启动GPU；独立数值正确性仍未知，不由HTTP成功推导正确性或G1/G2排名。

## 冻结输入与执行

- 原3B seed42 trace、500-adapter pool与真实远端一次性发布工件不变。
- 复用既有`adapter_coverage_v1`，与ServerlessLLM D758相同：500请求、500
  实际adapter IDs、68925目标输出tokens。
- view SHA：`4a9f63b08a600c40af7f5acd8066fb518aa21ed48ee8d5c75d7b3e2e8a51088e`。
- S-LoRA保持3B GQA/BMM、native HOST、四TP1 worker、RR、顺序启动；
  canonical token输入边界只修复D767的旧tokenizer重分词差异。
- BMM路径不等于7B packed BGMV分页机制，不宣称原论文所有机制均已覆盖。
- 共同60秒部署通知、真实准备/读取/传输/物化计入时延；请求中不打包。
- 原生token计数、adapter绑定、实际worker与GPU释放计量均复用既有检查。

服务72/80GiB、swap2GiB，独立回放3/4GiB、swap0，CPU隔离。启动使用原
150GiB/102GiB准入规则与独立watchdog；不会为过门槛修改资源或性能条件。
D768关闭的重复下载缓存已回收，唯一模型/工件/trace和历史结果不变。
新RAW为`results/ieee_tc/slora_qualification/d769_20261023`；小型输入绑定
备份于`paper_results/ieee_tc/baseline_audit/d769_slora_3b_coverage_binding/`。

## 解释与下一步

成功要求完整500条原生合同、目标token和500-ID覆盖，不要求SLO达95%才
保存。低SLO原样报告，不单独触发baseline优化；出现具体输入/计量/适配
错误才按既定边界修复受影响路径。完成后保存与释放资源，继续1000兼容、
完整W0/W1/W2和其余3B基线，之后再集中优化PrimeLoRA。
全部A1–A5、S1–S13及审稿证据交付保留；共享k3s不动，当前不画图。
