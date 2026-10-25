# D835：完整3B W2检验已完成绑定复用

2026-10-25，执行前登记；不表示已经启动、完成或达标。
唯一候选D834已通过452项不同CPU测试并备份fa56318e5cd3b77a9cdc8a8716534896c4b5395d。
原理、D116成员生命周期及既有负例/官方源码核查见D834说明。不重跑资格，
不叠加第二改动，也不把最小依赖验证当作端到端收益。

## 可证伪问题

已完整绑定的副本不再重复等待全域状态转换，是否能在相同完整churn负载下
提高联合TTFT/TPOT SLO、降低尾TTFT，并保持物理GPU预算？真正start仍需
同一把锁，因此不预测等量省时。新增snapshot计数只证明复用是否实际发生，
不代表因果节省时间。未触发、失败、退化一律保存。

## 保持的执行合同

- 配置与D832仅output/NVMe/HOST三个隔离路径不同；复用既有runner。
- 3B、完整4000请求、500逻辑adapter、seed42源trace和W2 rotation100映射；
  arrival、prompt、target及原生fixed_length_greedy_v1不变，不新增工件或负载。
- vLLM0.30、D725 profile、三卡上限、每runtime TP1、路由/驻留/admission/
  扩缩容规则及九公式不变；native_rpc_v1保持原start/finish确认顺序。
- D831文件可用与最终收尾所有权仍保留；首次绑定、退休、取消及HOST预算
  不变。实际run重查目标，不从历史attached状态推导新IO许可。
- 60秒共同部署通知、open-loop、自然闭环、1800秒保护保持。
- 真实remote沿用D78/D80只读发布缓存；读取/传输/竞争/本地物化计入时延，
  请求不打包，不注入模拟延迟，不改网络配置，不从旧时延事后扣除span。
- 依据冻结V1/单轮V2、D648共同SLO、D657 W2参考：联合至少3800/4000，
  U预算12083.063870235463GPU-s。console旧5000ms/模拟成本不作为主判断。
- D832开发参考：联合3683/4000=92.075%，U11308.666182523593GPU-s，
  P95TTFT2327.791816555ms。新版本只运行一次，不重跑同键筛选胜出。
- D711独立数值身份仍未知，N_correct=null/ranking_eligible=false；数量和
  内容绑定证据不冒充独立数值正确性。保留已知回放上的开发身份。

## 执行与保存

先13项入口/路径检查、明确路径备份、实际资源准入，再启动唯一tmux。
这13项仅覆盖新增执行键和共用分支，不重跑D834的452项，不增加短GPU回放。
使用run-experiment的已有受限入口；monitor-experiment按实际tmux/资源域/
请求日志监测。服务72/80GiB、辅助3/4GiB、既有CPU与swap限制不变；推理机
150/100GiB磁盘规则和独立远端规则保持。不操作共享k3s、其他项目服务或baseline。

raw：results/ieee_tc/p2_backend_qualification/d835_20261025/。
curated：paper_results/ieee_tc/p2_backend/d835_3b_completed_membership_reuse_full_w2/。
tmux：tc-d835-3b-completed-membership-reuse-full-w2。
所有历史目录保持原字节；运行中不清理/压缩/安装/调整远端。自然结束和实际
资源释放后，复用现有计量、来源及阶段分析，检验复用触发、完整SLO/U与
副作用，再决定保留或撤销；不因一次更好就宣称跨场景最优。

同版两模型主比较、A1–A5、S1–S13、CE/Pareto、两轮审稿文档、IEEE图表和
最终V2同步仍未完成；本轮不画图，整体目标active。
