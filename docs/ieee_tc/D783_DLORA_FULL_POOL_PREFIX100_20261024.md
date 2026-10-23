# D783：dLoRA 3B 完整池启动与100请求资格

2026-10-24。属于有界必要适配后的资格观察，不是完整W0/W1/W2性能结果。
原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d783_20261024`。

## 实际完成与证据边界

| 项目 | 实际观测 |
|---|---:|
| 计划/提交/正常终态 | 100/100/100 |
| 原生输出token | 17,369，全部逐请求匹配目标 |
| 已准备/两worker加载的adapter池 | 500逻辑ID |
| 本前缀实际服务adapter | 29 |
| 原生服务引擎 | 2，DP2/TP1 |
| 原生初始GPU放置 | 250/251，合计501份 |
| 物理GPU生命周期 | 818.521554742 GPU-s |
| 平均/P95 TTFT | 302,279.805 / 374,474.767 ms |
| 平均/P95 TPOT | 100.786 / 323.258 ms |
| 冻结共同阈值的联合达成 | 0/100 |
| E2E、TPOT及原生重算最大误差 | 0 ms |
| 服务内存峰值 | 77,310,402,560 bytes |
| memory.high / max / OOM / OOM-kill | 2368 / 0 / 0 / 0 |
| 服务swap峰值 | 718,372,864 bytes |
| 主机最低MemAvailable | 35,177,693,184 bytes |

两worker实际完成500个PEFT adapter CPU加载，分别初始化1173/1143 GPU KV
blocks及4681/4681 CPU KV blocks。保持原500池、CPU KV配置和原生
EngineManager/PERIOD_MIG；没有缩池、缩减KV或替换为前端轮询。
D782 CPU零模块只读共享表示是本轮唯一新增运行适配，GPU表示和原生决策不变。

该100前缀全部在长准备阶段积累。共同通知到GPU分配约149.103秒，到manager
ready约429.123秒；业务仍从通知后60秒开始，未按ready平移。平均提交滞后
仅6.331ms，平均提交后dispatch等待264,166.446ms，平均service TTFT
38,107.028ms。该启动与积压条件解释了很差的前缀服务质量；不能由此前缀
推断完整W1/W2吞吐或宣称dLoRA机制必然劣于PrimeLoRA，也不因此搜索基线参数。
共同内存包络发生实际high与swap，属于所测性能条件，不能写成无内存压力。

原生request/model-ID、prompt/target、engine-path与实际GPU进程完成关联，
两GPU实际exit/release闭合。17,369原生token与目标吻合不是独立模型数值正确性
证明，该状态仍为未知；29个served ID也不等于500-ID功能覆盖。
仅配置PERIOD_MIG不证明迁移触发，本轮没有把配置当作迁移收益证据。

## 远端与数据保护

完整500池在共同通知之后从真实远端读取既有预发布gzip：线上
1,160,474,851 bytes，本地核验19,482,573,296 bytes。请求路径打包次数为0。
这些是部署准备传输，不能按每请求引用重复计费；各阶段时间不重复相加。
远端三个本次所属服务已精确关闭；共享k3s未修改。实际service、publisher和
watchdog均返回0，GPU上下文与服务资源域均已释放。

原始结果、native Ray日志、远端transfer/monitor和resolver凭据保留。只在
完整内容校验后，使用已有D532清理规则回收有原池保留副本的本轮下载objects；
具体释放字节和保护文件核查见同目录storage/cache_cleanup回执。

## 离线分析适配

复用主仓库`audit_d402_serverlessllm_path.py`和baseline原汇总程序，增加显式
`dlora_native_manager_v1`，不用S-LoRA轮询合同套用dLoRA。
核验单manager、两原生worker、完整池与已记录GPU租期；复用既有原生响应、
共同准备窗口、内容和SLO校验。工作配置仅允许原服务配置加准确的pool路径/
subset/limit三项派生参数，不忽略配置差异。

首次离线分析因旧汇总不识别dLoRA而停止；第二次因尚未识别上述明确配置派生
而停止；修正离线消费者后第三次分析成功。三次记录均保留，GPU实验只运行一次，
没有改变已执行服务、输入、生成或结果。CPU回归测试和实际审计见analysis日志；
不将单元测试作为性能或数值正确性证据。

## 下一主线

该100前缀资格可复用，不重跑。按已批准顺序补500-ID实际服务覆盖、1000请求
兼容，随后只做dLoRA限定3B W1/W2。保留Loquetier/HydraServe 3B、PrimeLoRA
全对照优化及同版本Full、A1–A5、S1–S13、成本/CE与审稿文档/最终图表。
整体目标仍active，G1/G2最优性尚未完成。图表按用户修订继续延期。
