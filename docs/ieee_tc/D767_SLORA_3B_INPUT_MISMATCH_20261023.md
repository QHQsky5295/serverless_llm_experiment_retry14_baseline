# D767：S-LoRA 3B首100请求资格的输入兼容失败

2026-10-23。沿用已备份的3B native BMM配置，未改原生调度、LoRA策略或权重。
本次是受影响路径资格，不是主表性能实验，也不属于baseline性能调参。

## 实际结果与可归因范围

- 100计划、到达、提交与终态；99条通过原生合同，1条失败：`req_00028`。
- 该请求在首token绑定核验处失败。离线复用相同输入生成函数与原生分词器，
  100条中只有该请求不一致：共同输入89 tokens，原生输入91 tokens；首次差异
  在零起点34位置。原生环境为Transformers 4.34.0/tokenizers 0.14.1。
- 这是同一prompt在两个分词环境中的输入工作量差异，不得将失败算作S-LoRA
  原生算法弱于PrimeLoRA的证据。没有删请求、改目标长度、放宽检查或重跑择优。
- 4个原生worker已结束，GPU实际释放；全部自有远端服务停止，共享k3s未动。
  生命周期物理GPU时间为1230.8150130887516 GPU-s，只作本次资格消耗记录，
  不进入4000请求排名。独立数值正确性仍未知，`n_correct=null`。

共同分析器的static-native receipt路径还假设所有response均有metrics，遇到
本次保留的HTTP 500触发KeyError。分析失败日志保留；没有将这次分析失败
伪作完整审计通过，也没有从99条成功请求构造全量SLO结论。

## 资产与下一步

原始证据：`results/ieee_tc/slora_qualification/d767_20261023`；小型封存：
`paper_results/ieee_tc/baseline_audit/d767_slora_3b_entry/gpu_prefix100`。
原资格配置/52项CPU检查、原生代码历史和首100输入身份仍保留。

下一步只处理必要输入兼容：核查历史与官方输入接口，保持共同prompt及确切
token序列，不升级/替换原生推理核心，不引入回退；补受影响路径验证。此前
完整7B、vLLM 3B、ServerlessLLM 3B结果不因此重跑。3B S-LoRA后续覆盖、
1000兼容、W0/W1/W2及全计划范围不删减。已逐字节核对并回收4000个重复
下载文件，释放19497328640B，剩余163823943680B；唯一模型、工件、原始
失败和旧投稿结果不删除。新实验启动时仍重新检查磁盘/内存余量。

已核查固定官方commit的[HTTP入口](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/api_server.py)
和[manager](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/httpserver/manager.py)：
文本在manager中重新encode，再将token IDs送至原生router。当前本地源仍
保留这一边界；这与CPU复现一致。一个待验证的最小适配方向是显式传递公共
输入token IDs，同时保留文本hash和原生长度/ID核验，避免跨版本重复分词；
尚未实现或验收，不能据此宣布修复完成。不得静默回退、按请求删减token或
改变原生调度/LoRA计算。既有7B未受影响的结果无需自动重跑。
