# D800：Loquetier 3B完整W0（配置登记，结果待完成）

D799已经完成500-ID功能覆盖并保存生成/绑定/计量证据。D800直接运行既定
完整W0，不再增加100/1000请求门禁，不因D799覆盖视图SLO低而搜索baseline
参数。原生模型、SMLM、四静态分片、batch4、顺序启动与PEFT owners全部不变。

## 冻结输入与实现

- 复用原3B seed42 trace，4000请求；`trace_prefix_v1`、`replay_profile=W0`。
- 500逻辑adapter universe，原主trace实际出现132个，目标输出458,224 token。
- `view_sha256=0d998f48a006d638dda9b8d714f667fe3a87f4d2c88044ce19a816b308494a37`，
  与D771 S-LoRA 3B W0逐条source JSON及输入身份一致，没有新造trace或权重。
- `native_3b.json`与D799逐字节一致，SHA
  `3b5a0274271a95965317bc6bf803728e4eb246d6026ab4338bb640dfc7bff800`。
  引用同一D799 measurement view与静态计划、D797模型专用kernel，未重新构建。
- 复用D799的37项CPU测试、一次完整池覆盖与D798有界模型数值检查；当前只
  验证新W0输入绑定和既有来源SHA，不重复不受影响的资格。
- 真实远端D80已发布gzip，部署通知后准备完整静态池、无请求打包或注入等待。
  共同60秒准备、原生固定长度greedy、原生token时间及完整GPU生命周期不变。
- 保持`formal_run=false`、开发观测身份和未知全负载数值正确性字段；不在运行
  前或仅凭HTTP成功宣称完整正确性/正式资格。SLO仍用D648冻结3B共同阈值，
  G1/G2与单轮V2不改，0SLO也完整保留，不作为排除baseline的理由。

## 运行与保存

只复用既有gated-launch/native-http-service、独立watchdog及精确所有权清理。
长任务使用唯一tmux `tc-d800-loquetier-3b-w0`，输出
`results/ieee_tc/loquetier_qualification/d800_20261024/`；不覆盖D799或旧投稿。
原生准备和请求按真实时间进行，不按就绪时刻平移业务。推理服务72/80GiB、
swap2GiB与辅助3/4GiB、swap0包络保持。磁盘150/100GiB门槛继续执行。

上一轮重复缓存已回收；登记时空闲约150.05GiB，启动前重新检查。运行中不
扫描/清理/压缩/构建/推送或调整远端，终态之后实际释放、校验、保存，再
推进同配置W1/W2、HydraServe3B和完整基线比较。其他项目k3s不修改。

本文件登记执行，不包含未产生的性能结论。原始结果无覆盖；失败与安全事件
保留。后续PrimeLoRA优化、同版Full/A1–A5/S1–S13、成本/CE/Pareto、两轮
审稿文档与最终图表/V2交付不删减。
