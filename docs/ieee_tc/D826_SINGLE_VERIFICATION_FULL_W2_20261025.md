# D826：单次最终内容校验候选的完整3B W2评估

2026-10-25，运行前登记。D825已完成446项不同CPU检查与实际32MiB组件，
并备份4a7c979ba2efe1cc47bc9d8f915df7896181e614。完整假设、官方来源与
失败历史见D825_SINGLE_DESTINATION_VERIFICATION_20261025.md；不再重复
CPU资格或新增短Full。D826是一次新的完整执行键，不是D817同键重试。

## 不变条件与唯一候选

仅采用D825受管HTTP的owner_publication：去掉解包流重复payload SHA；
保留原owner锁内的最终目标内容SHA、真实容量、取消和原子确认。压缩包
验证、真实传输、所有必要本地工作均计入原生用户TTFT，不事后扣时延。
不启用D822的共享压力通道，保持native_rpc_v1默认；不移锁、不增并发、
不改变九公式或控制阈值。默认HTTP及baseline路径无性能修改。

配置与D817仅三处独占输出/NVMe/HOST路径不同。max3、1初始实例、8并发、
vLLM0.30.0、已有graph/descriptor、D725 service/preparation profile及在线
EWMA、5s demand window、1800s保护、共同60s部署通知均保持。W2复用既有
4000请求完整源trace和rotation100 adapter map，不复制工件、重抽负载或缩池。
source seed42、known-workload development observation，不能改名为留出验证。

评价遵守冻结V1（SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22）
与单轮V2（SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d）。
共同3B SLO D648、Resident D657参考不变；联合≥3800/4000才达阈值，
G2预算12083.063870235463物理GPU-s不变。CE不作准入或配置选择依据。
D711独立数值核查限制继续保留，N_correct不因HTTP或token数量正确而补为4000。

## 评价与停止规则

前序D817：联合3596/4000、物理U11283.832798304502、P95TTFT2693.5871988534927ms。
这不是达标对照。必须报告本轮完整的请求/原生token/时间一致性、联合及
边际SLO、物理生命周期U、尾TTFT、TPOT、准备/加载分解、失败和资源事件。
不以局部组件下降比例投影完整请求，不把自然闭环不同层命中组当因果对照。

仅执行一次。无新证据不重跑本键，不筛启动/种子/请求、不更换SLO或预算。
若总体退化，保留原始记录并否决，不因局部SHA减少就继续扩大W1/W0。
若改善，仍按同版W1/W0和完整指标继续验收，不把旧不同版本场景拼成达标。
即使本轮过95%，独立数值限制与未完成两模型主矩阵也不自动闭合。

## 资源、复用与产物

复用现有run_d293完整分支和user scope入口；一重型、服务72/80GiB、
swap2GiB及CPU4–23/28–47，外置3/4GiB、swap0及CPU2/3/26/27。
运行前实测MemAvailable与150GiB磁盘门槛，预测新增峰值32GiB；运行中100GiB
磁盘底线及既有内存watchdog不变。远端继续D80已发布服务及独立保护阈值，
只管理本项目服务，不改共用k3s。运行期间不清理/压缩/哈希扫描/推送。

原始：results/ieee_tc/p2_backend_qualification/d826_20261025/3b_max3_full_w2/。
汇总：paper_results/ieee_tc/p2_backend/d826_3b_single_verification_full_w2/。
完成后先释放和校验、保存全部失败/结果，再判断后续。目标未达前不绘图。
147旧投稿保护和3处用户修改不变；启动前备份明确路径，不stage用户manifest。
完整同版两模型M1/M2、A1–A5、S1–S13、成本/CE、审稿文档和图表/V2同步保留。
