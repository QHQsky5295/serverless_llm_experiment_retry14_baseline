# D832：有所有权的文件就绪重叠——完整3B W2

执行前登记，不代表已启动或性能接受。唯一候选为D831：实际文件IO及清理
已join后，先在owner锁内取得真实读租约，原有顺序的压力finish仍由请求拥有，
可以和原生加载/生成重叠，但成功终态必须join，迟到错误使请求失败。
435项不同CPU用例已通过，首两轮fixture失败也已保存；代码检查点
beb6a1bba213b33df86711f83a13fc4679adac07已推送并核验
backup/d831-owned-file-readiness-20261025。不是最终V2同步。

## 唯一待证伪问题

完整churn负载中，这一真实数据依赖调整能否提高共同联合SLO、改善P95 TTFT，
同时保留物理GPU预算和完整E2E计量？D830的条件等待均值不从历史结果中扣除，
不预测达标条数。保留全部失败、退化和机制未触发，不保证候选必胜。
历史否决方向、官方vLLM/Python及ServerlessLLM依据见D831说明；不重复
并行ACK、共享memfd压力通道、移hash锁或清理优化，不叠加第二候选。

## 受控合同与验收

- 与D829配置仅output、NVMe、HOST三个隔离路径不同；后端0.30、三卡上限、
  单卡runtime、生成、profile、路由、规划、admission及扩缩容配置不变。
- 完整4000请求、seed42既有源trace与W2 rotation100映射；不重造权重/负载。
- 真实remote、既有只读发布对象、真实传输计时；不在请求内打包，不注入等待。
- 共同60秒通知、open-loop、自然闭环、1800秒保护、冻结SLO/预算不变。
- 按V1与单轮V2：共同TTFT/TPOT至少3800/4000；物理GPU-s与预算分别报告。
  D829参考为3638/4000、U11246.510437、P95TTFT2514.899193ms，尚未达标。
- 核查file_preparation_settlement实际结束、迟到失败、原生首末token、E2E
  恒等式、真实文件/原生引用和GPU释放；不将重叠活动重复加和。
- D711独立数值身份仍未知：N_correct=null、ranking_eligible=false。CPU路径、
  原生输出长度或内容绑定不冒充独立数值证明；保留已知负载开发身份。

## 启动与保存

复用已有完整W2 runner、远端启动/监控/按身份释放；先11项入口/配置检查、
明确路径备份、实际资源准入，再启动唯一tmux。没有额外短Full/GPU微测。
服务72/80GiB、外置3/4GiB、推理机150/100GiB与防OOM规范不变。435项
CPU验证直接复用。147旧投稿及3处用户修改不变，不操作共享k3s或其他服务。

raw：`results/ieee_tc/p2_backend_qualification/d832_20261025/`。
curated：`paper_results/ieee_tc/p2_backend/d832_3b_owned_file_readiness_full_w2/`。
tmux：`tc-d832-3b-owned-file-readiness-full-w2`。
一旦启动，配置冻结；运行期间不清理/压缩/安装/修改远端。自然结束、确认
释放后复用完整分析，补本候选覆盖记录，再接受或撤销。不重跑同键择优。

完整后续范围仍为同版7B/3B W0/W1/W2、M1/M2、A1–A5、S1–S13、
CE/Pareto、两轮审稿文档、IEEE图表及最终V2同步；目标active，不提前画图。
