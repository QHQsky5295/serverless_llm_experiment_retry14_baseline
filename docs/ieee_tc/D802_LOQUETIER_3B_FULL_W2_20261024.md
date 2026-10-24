# D802：同配置Loquetier 3B完整W2登记

## 科学问题与输入

完成W0/W1后，执行同一冻结Loquetier 3B工作点的高频adapter轮换场景。
使用既有3B seed42的4000请求、相同arrival/prompt/target，只采用已冻结
rotation100映射；不再生成trace、adapter权重或第二个完整工件池。
W2与D773 S-LoRA输入逐条核对；实际497个逻辑ID、458224目标token，
不是500个独立训练模型的证明。

- 源trace SHA：`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`。
- 既有W2 map SHA：`28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23`。
- 预定义view SHA：`5558de43ede0e2cf49c7478dcfb94640e7a8516406f53766fabfdf524bf91174`。
- native配置SHA：`3b5a0274271a95965317bc6bf803728e4eb246d6026ab4338bb640dfc7bff800`。
- 共同3B SLO SHA：`d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`。

## 保持不变与复用

复用D799的500-ID完整功能检查、37项CPU测试、D798有限数值对照，以及
D800/D801的原生完整运行路径。保持四个静态125-ID分片、SMLM、batch4、
顺序初始化、PEFT owners、D797配置驱动GQA/RoPE内核。不因W1低SLO换
参数、绕过共同阈值、修改对手策略或额外加短回放。

共同部署通知业务前60秒，真实远端D80预发布gzip、cold per-run本地缓存；
部署准备实际读取/传输/加载计入时间线，原生缓存合法保留。请求路径不打包，
不进行远端维护、扫描、压缩或额外压测。仅在新独立campaign下运行一次。

服务72/80GiB、swap2GiB、CPU4–23/28–47；辅助3/4GiB、swap0、CPU2/3/26/27。
复用原gated launcher、watchdog及精确所有权释放。推理机启动150GiB、运行
100GiB和主机内存护栏不变；D801重复缓存安全回收后才进入本任务。

## 验收与后续

按冻结V1/单轮V2，保留全部4000 offered、实际终态/原生token/adapter绑定、
TTFT/TPOT/E2E分解、物理GPU分配至释放和共同联合SLO。未知的完整负载
独立数值正确性继续保留未知；必要路径验证不自动升级为完整数值证明。
低SLO如实封存，不筛选重跑、不称未搜索的工作点为Loquetier全局最优。

本文件是运行前登记，不是完成或达标声明。运行后释放→核验/保存→重复
缓存回收→结果备份，然后完成隔离HydraServe 3B的必要模型适配/场景，再
作全基线比较与PrimeLoRA优化。其他项目k3s不修改；同版Full、A1–A5、
S1–S13、成本/CE、两轮审稿文档、图表及最终V2全部保留。
