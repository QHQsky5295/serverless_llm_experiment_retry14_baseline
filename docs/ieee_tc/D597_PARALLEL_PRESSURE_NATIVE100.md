# D597：并行压力确认的原生最小验证

2026-10-17。D595 已推送 `8639ce510fa5fe0d20adfe93e922a280751acf77`，
31项聚焦/368项完整CPU回归通过。当前只验证这一候选，不增加第二项优化，
不改九公式、共同SLO、预算、模型参数或准备窗口。

## 诊断范围

- 使用 D550 的既有 W1 前100请求、D554完全相同的模型/控制配置。只换独立
  输出、NVMe和HOST缓存路径；不生成权重、trace或另一工件池。
- 对原计划“W2前100”作实施层修正：现有 W2 loader 明确要求完整4000请求，
  不能直接传count=100。此次不为短诊断改变已验证的输入合同，复用W1前缀；
  随后的完整W2才检验高churn收益，短前缀不能替代其效果结论。
- 复用D293 runner、D550原生计量/闭环收尾和分析，复核最终D595源码SHA。
  完整W2的冻结adapter-map、source/view SHA另行验证，后续不重新抽样。
- 原生token、adapter绑定、dispatch快照、E2E分解、物理allocation/release、
  双端transfer ID/字节/SHA和远端无请求打包均需合格。
- 保留压力事件的policy、真实owner数、原生start/finish确认及结算状态；
  迟加入成员按实际时间报告，不能当作起始参与者的错误顺序。没有独立RPC发送
  时间戳，不能从累计RPC时长捏造精确重叠量；D595事件屏障测试验证并发语义。
- 若前缀没有多owner则如实记未触发，不反复跑短前缀凑数；完整回放保留该检查。
  数值独立正确性仍未知；100请求不是4000请求合格或G1/G2已完成。

## 依据与主线

D594已测W2远端请求的压力确认串行等待，D595只移除独立owner之间非必要的
先后依赖。2026-10-17再次核查
[vLLM 0.30 collective RPC](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/multiproc_executor.py)
及[Python gather/shield语义](https://docs.python.org/3.12/library/asyncio-task.html)；
它们支持广播后集中确认及取消时保留所有权的实现依据，不证明本系统性能必升。
历史D516/D529退化方向不重复。共享k3s不访问/不修改，HydraServe隔离适配及
7B三场景已完成；不恢复无界基线调参。

启动前D596仅将已关闭D533/D512/D519的main_outcome无损压缩，源码/原始SHA、
两次回读SHA及逐文件恢复映射保留，释放649056256B；未删除唯一证据或旧投稿。
推理机150/100GiB门槛保持，内存/CPU限制由既有runner实际执行。

下步：完成本原生验证→新键完整W2→同配置W0/W1及G1/G2判断。3B、限定dLoRA、
A1–A5、S1–S13、成本敏感性、审稿设计文档和最终图表全部仍在完整任务中。
