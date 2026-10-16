# D589：Loquetier 固定容量配置的完整 W2

2026-10-16，运行前登记。D587 W1已封存，D588存储维护已备份到
`a1f687be2c4e2b82d69f95341e3701645f2ab141`。本轮只补齐既定churn场景，
不继续基线调参，也不因W0/W1未过95%而隐藏结果。

## 问题和不变条件

检验相同Loquetier部署在rotation100下的服务质量、等待与生命周期GPU占用。
复用D557及其他W2对照的既有4000请求视图，不生成新trace、权重或工件池。

- 源trace SHA：`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。
- W2 map SHA：`48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`。
- W2 view SHA：`5734bd88587c75dacc0d5c7534ad5b4d5d2a67aa0980f9b203dd87135d3adce7`。
- 4000请求、497个实际adapter、458224目标token，完整500-ID池；原arrival/
  prompt/target不变，已有adapter映射采用rotation100。
- native配置SHA仍为`455ef31c293e5f55e35397cfaccfd167b422bc34d31be3c95a876c0ec2922fc6`，
  与D585/D587逐字节一致：原生SMLM、FP16、四个125-ID静态分片、batch8、
  context1024、FIFO、顺序启动及已验证的表示生命周期适配均保持。
- HTTP配置SHA：`d1009852786f306bbb4a1c74b105b12091d34cea4f1f5fd2ffa7942a4b04d966`。
  相对D587仅改既有W2视图/映射/源路径与场景身份；服务及生成参数不变。
- 共同业务前60秒部署通知、真实远端D78/D80已发布gzip、固定长度greedy、
  原生token计量、原生adapter绑定、完整物理GPU生命周期不变。请求中不打包。
- 保留498零权重/2非零、4种独立权重SHA及数值证明未知的局限；不能由
  逻辑ID或HTTP成功声称独立数值正确。D587旧新349条输出差异继续保留。

## 资格和安全

复用原有runner、管理与分析入口。按当前源码SHA核验并复用86项在线测试、
69项分析测试、D567完整500-ID功能覆盖和D569的1000请求兼容结果；本轮只
新增W2输入/人口合同检查，不重复无变化GPU资格。
资格已通过，记录见本轮qualification_final.json。第一版离线检查构造的contract
遗漏W2既有tc_adapter_rotation字段，被原分析器拒绝；失败源码、错误和实际关闭
记录保留。第二版从已冻结W2源行带入该字段后通过，未修改负载、生成器、分析器
或服务实现；两次离线检查均无GPU/远端启动，不属于性能重跑。

新结果目录`results/ieee_tc/loquetier_qualification/d589_20261016`，
私有cold缓存`/tmp/tc-d589-7bl`，tmux `tc-d589-loquetier-full-w2`。
先备份来源，再执行新鲜启动门槛：磁盘至少150GiB、空服务MemAvailable至少
102GiB，预测新增峰值20GiB；运行磁盘底线100GiB。服务72/80GiB、swap2GiB，
外置3/4GiB、swap0与CPU隔离不变。实际worker受限，不修改无关k3s。

仅一个重型任务。测量期间不构建、测试、批量扫描、压缩清理、推送或修改远端。
采用实验执行技能复用入口/tmux，资源技能核查实际限制，备份技能保护来源，
监测技能依据当前进程和终态证据，不因观察超时重新启动。

## 评价、保存和返回主线

冻结V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`，
单轮V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`，
D394 SLO SHA `2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3`。
保留formal_run=false/开发观测身份，单轮、不补CI，不按结果选择重复。

实际结束后：核验GPU和服务释放，关闭确切归属远端服务，保存双端传输证据，
用原分析器full4000_w2模式及既有合格Python环境处理实际success/failure；
逐请求核查输入、token、绑定、时间恒等式、资源积分及机制范围。与同W2输入
进行对应，保存失败、数值未知和未达标点，再备份。95%不是基线纳入门槛。

随后补齐剩余必要7B对照，汇总后集中优化PrimeLoRA。不得因本点低SLO再开
Loquetier参数搜索。全部3B、dLoRA限定比较、G1/G2、A1–A5、S1–S13、成本/CE、
两轮审稿文档和最终图表仍需完成；接近目标前不画图。
