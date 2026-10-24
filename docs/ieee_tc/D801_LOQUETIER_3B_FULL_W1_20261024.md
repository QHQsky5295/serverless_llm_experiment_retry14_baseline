# D801：Loquetier 3B完整W1（配置登记）

## 科学问题与不变项

D800的原W0已完成4000请求并保存。D801回答同一部署面对既定突发负载时，
请求排队、联合SLO和物理GPU时间如何变化；不是baseline优化或重跑W0择优。
W1沿原4000请求顺序分8×500，每段内部到达间隔减半，段间30秒；不等待
实际drain，不人为强求八次扩缩容，不改变adapter或目标输出。

- 原3B seed42源trace SHA：
  `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`。
- W1 view SHA：`8a7fa829c4902a761efcf58c765580cbdb5e0f367e3c2245ea43417409e9494f`，
  与D772 S-LoRA W1逐条source JSON和输入身份一致。
- 500-ID universe中本视图实际出现132个，目标输出458,224 token。没有新生成
  权重或完整trace，只复用既有参数化视图。
- native配置与D799/D800逐字节相同，SHA
  `3b5a0274271a95965317bc6bf803728e4eb246d6026ab4338bb640dfc7bff800`：
  四静态分片、batch4、原生SMLM、顺序初始化、保留PEFT owners及D797专用
  Llama3内核不变。HTTP配置SHA
  `5a1493d2fa95637ae8419c71b8958d1fd916fce1971f506ae4e325039224f280`。
- 复用D799完整500-ID功能覆盖、37项CPU测试和D798有界数值检查；不添加
  100/1000请求短门禁。D800的低联合SLO不触发baseline配置搜索。
- 共同60秒部署通知、真实D80预发布只读gzip、无请求打包/注入等待、固定长度
  greedy、原生token计时和完整物理GPU生命周期保持。3B SLO仍为D648冻结值。
- 保留开发观测、`formal_run=false`、全负载独立数值正确性未证明；未运行前
  不填写性能结论，不因未来完成HTTP响应自动升级正确性证明。

## 登记、资源与证据

原始结果位于`results/ieee_tc/loquetier_qualification/d801_20261024/`，
tmux唯一名称`tc-d801-loquetier-3b-w1`。使用现有gated-launch、原生服务、
独立watchdog、精确所有权关闭和后置analyzer，不新增一套执行框架。
服务72/80GiB、swap2GiB，辅助3/4GiB、swap0，既定CPU分离和150/100GiB
磁盘限制不变。D800重复缓存与已关闭D726日志已处理；启动前再检查实际余量。

首次CPU登记已通过输入/配置校验，但在写preparation.json前因读取D800的
`tests`字段失败（实际名称为`tests_reused`）。没有启动GPU或远端。
保留失败脚本SHA和prepare_error.json；finish_prepare.py只读取准确字段、
重验现有配置而不覆盖、完成唯一登记。两次CPU辅助域均关闭；这不是两次
性能尝试或系统退化，不据此重跑任何资格实验。

运行期间不修改服务/配置、扫描清理、压缩、构建、推送或维护远端。真实终态
后释放→核验→保存→安全回收重复缓存，接着同配置W2。若有失败保留实际
4000分母和失败证据，不按胜负调整时限或SLO。

之后仍是隔离HydraServe3B（不改他项目k3s）、完整baseline比较、PrimeLoRA
有证据优化、同版Full/A1–A5/S1–S13、成本/CE/Pareto、两轮审稿文档、图表
及最终V2交付；不以W0/W1局部完成缩减范围。
