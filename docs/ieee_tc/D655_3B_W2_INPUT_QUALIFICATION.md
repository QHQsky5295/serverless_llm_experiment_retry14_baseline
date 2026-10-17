# D655：3B 高频轮换的共同输入资格

2026-10-18。完成 D653/D654 的 3B Resident W0/W1 后，沿完整 PLAN 推进 W2。
本项只核验输入，不是新的性能测量、算法优化或独立数值正确性证明。

## 受控变量与原有数据复用

复用 D553 的确定性索引变换与已有 FrozenReplayPlan；不改生产代码、不重新
生成 trace、prompt、权重或整池工件。3B 原始 trace SHA 为
`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`，
原 W0 view 为 `0d998f48a006d638dda9b8d714f667fe3a87f4d2c88044ce19a816b308494a37`。
使用该模型的实际 selected_adapters 顺序及逐行来源，不套用 7B 映射。

源热点宽度48、前移步长12、相邻窗口75%重叠。请求序号从零开始，源adapter
位置为 p_i；恢复 k_i=(p_i−12 floor(i/500)) mod500，再取目标位置
(k_i+12 floor(i/100)) mod500。全部4000条的原局部rank均在0–47。
只把轮换间隔500改为100；不重新抽样rank，不改到达、prompt或目标输出长度。

核验事实：4000请求、458224目标token；实际unique ID从132变为497，3900条
改变adapter。新索引1032987字节，SHA
`28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23`；
W2 view SHA `5558de43ede0e2cf49c7478dcfb94640e7a8516406f53766fabfdf524bf91174`。
索引在 `paper_results/ieee_tc/inputs/20261018_3b_w2_adapter_map_v1.json`。
497个逻辑ID不等于497组独立训练权重；既有内容关系原样保留。

## 路径核验与边界

- PrimeLoRA实际输入读取、共同HTTP入口及native配置校验接受同一完整W2视图。
- 使用既有3B tokenizer，4000条canonical prompt SHA、native input-ID SHA、
  target与W0逐条相同；request ID、到达时间及源生成长度不变。
- 请求adapter与body LoRA名称同步更新，domain仅取目标静态元数据；逆向撤销
  映射注释/adapter/domain后原始行完全相同。prompt不随domain改写，因此这是
  受控churn压力输入，不是新语义质量数据集。
- analyzer仅接受full4000_w2，不接受冒名W0/W1；错误map/source/pool、短回放
  或冲突profile由既有检查拒绝。7项针对性CPU检查通过，无跳过。
- D649的89项native服务资格按全部来源SHA及用户修改SHA复用，不重复运行。
  D648共同3B SLO、D649原生配置不变，不进行baseline参数搜索。
- 没有启动GPU/远端服务；不能从输入资格推断性能或数值adapter效果。W2换了
  adapter，不能把W0/W1输出序列直接作为其数值正确性参考。
- 原源trace及147个旧投稿保护文件不变；两处baseline用户修改及manifest用户
  修改均不stage。CPU任务在既定3/4GiB、swap0和保留CPU集合中完成。

RAW：`results/ieee_tc/main_comparison/d655_20261018`；小型封存证据在
`paper_results/ieee_tc/main_comparison/d655_3b_w2_input`。
本检查点同时归档D654安全回收及publication关闭回执，不覆盖旧结果。

下一步同原生配置、共同真实远端和独立冷下载目录进行一次完整3B Resident W2；
先核查磁盘/内存/GPU与备份，再启动。W0/W1不重跑；三场景资源参考仍待单独
冻结。3B Full/其余baseline/dLoRA、同版本7B消融、A1–A5/S1–S13、审稿说明及
最终图表继续保留，PrimeLoRA尚未跨全部场景最优。
