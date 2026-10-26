# D883：替换代价共同解析的完整 3B W2 检验

本轮只执行 D882 已通过 322 项检查的候选，不新增生产机制。
研究问题是：保持同一新鲜物理图、定价与预算语义，消除两层替换代价的
重复验证/解析，是否改善完整 churn 工作负载的联合服务达成与资源占用。
组件微测不代替系统结果；本文件在启动前登记，不预先宣告收益。

## 比较与冻结条件

- 参照 D870：3773/4000 联合时延达标（94.325%），P95 TTFT
  2006.444557 ms，U=11931.565094 GPU-s。共同预算12083.063870 GPU-s。
- 复用同一3B、4000请求、500逻辑adapter、seed42原始输入和W2 rotation100
  映射；fixed_length_greedy_v1、真实远端已发布gzip交付不变。
- 配置相对D870只更换output/NVMe/HOST三个独占目录；仍是min1/max3、
  D725初始化、同一在线控制、原生descriptor与isal解压，不启用private CPU
  construction或整图descriptor。不新建profile、不加短Full或profiling。
- 代码父检查点43934903a32987523bdd6a1864ee43fbc7a18668；本轮记录实际
  启动commit。D882源SHA/等价性与既有native资格复用，不重跑整套检查。
- 冻结V1/单轮V2共同生效；至少3800/4000联合时延达标，不用在线5000ms
  代替评价阈值。N_correct仍需独立数值证据，未知不能填4000。
- 这是已知工作负载开发版本的一次完整观测，非独立留出、无重复CI；
  无论改善或退化都保存，不为取得胜出重复同键。

## 安全与出处

复用既有run_d293入口、资源域、外置回放/watchdog和整组释放；共同
72/80GiB、swap2GiB服务包络，外置4GiB，CPU分离。推理盘150/100GiB
与剩余增长门槛不变。真实远端仅管理本项目两个工件服务和独立监控，
不改共享k3s，不重新发布缓存，不在性能测量期间清理、压缩、哈希扫描或构建。

raw为results/ieee_tc/p2_backend_qualification/d883_20261026，curated为
paper_results/ieee_tc/p2_backend/d883_cost_fusion_full_w2。旧147投稿项和
用户manifest继续保护。明确文件提交到备份ref；最终V2分歧不force push。
run-experiment/monitor-experiment沿用已有入口与实际进程身份监测；
github-sync用于启动前的测试、SHA、凭据检查和可回退备份。

## 顺序与范围

运行结束后先验证实际释放、4000终态、原生token、时间线、物理GPU区间、
真实交付与准备机制回执，再离线按共同SLO/预算比较D870。没有结果前不画图，
不改阈值或模型配置。全部双模型同版本W0/W1/W2/G1/G2、剩余baseline/dLoRA、
A1–A5、S1–S13、数值正确性、两轮审稿图文与最终V2交付保持开放。

启动与完成状态以原始回执及EXECUTION_STATUS为准；本登记不代表已经运行。
