# D542：S-LoRA 7B W1 生命周期计量修复后的单轮测量

2026-10-15，启动前登记。D540 的4000请求已完整结束，但公共退出计量在
高编号pidfd上报错，缺一条真实释放记录，故不能用于完整资源比较。失败已
封存并推送31526e7；不由idle采样猜测GPU-s，不覆盖旧请求记录。本轮是
R3受影响点修复后的新attempt，不是因SLO不佳重跑寻找胜出结果。

## 唯一执行变化与复用

公共计量修复D541（0e5f25e5107971ab1b2c116dfdb34d4708c5edf7）将原pidfd
的select等待改为poll，保留共同deadline、全部worker退出、幂等和真实GPU
释放检查。原错误已用真实CPU子进程复现，修复后53项CPU检查通过；不再
重复这些检查。详情见D541_PIDFD_EXIT_MEASUREMENT_FIX.md。

S-LoRA原生策略、DP4/TP1、round-robin、顺序启动、packed HOST、paging/swap、
batch与容量配置均不改变。复用D540的51项原生接入检查与同一测量源码视图；
baseline仓库保持bde215460acb9e850efa479a89bda6fc39a2bd7d。
本轮两份配置所有JSON字段与D540相同，仅增加文件末尾换行：

- native SHA：c4725bc717a0fc21afc5302a3fd32223fe9ae837224e784affa42b8ba73024ce。
- HTTP SHA：2319ab246f71eeb34d3246b0b368105a476659dfe87c050ec1c1ad2e7877d9bd。
- 测量视图SHA：59399af84b2f9515fc2bacea94987dde882f8b381992e4a8a837f7c788ba6fcf。
- W1视图SHA：103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3。

4000请求、132个实际请求ID、458224目标输出token不变；部署通知后准备完整
静态500-ID池，不以未来请求筛成132个。498零权重/2非零权重的原池限制继续
披露；原生绑定证据不冒充独立数值证明。fixed_length_greedy_v1及通知后60秒
开始业务不变，W1八阶段与30秒间隔不变。无fallback，不移除早期等待。

新原始目录为results/ieee_tc/slora_qualification/d542_20261015，新冷下载缓存
为/tmp/tc-d542-7bs。既有模型、工件、trace、来源视图直接复用，不生成新数据。
仅在启动前记录现有Triton缓存；不清全局缓存或额外预热GPU，不假定与D540
的全部编译初态相同。开发身份41、formal_run=false保留。

## 安全、计量与顺序

复用已有受限runner、tmux与独立watchdog。服务72/80GiB、swap2GiB；辅助
3/4GiB、swap0；CPU集合及102GiB空载内存启动条件不变。预测新增峰值20GiB，
启动磁盘150GiB、运行停止100GiB门槛不变。D541已定点回收D540重复下载对象，
不重复清理；启动前重新检查当前资源。147项旧投稿文件和用户修改保持不变。

远端复用D78/D80只读gzip，真实传输/竞争/本地物化计入时延，请求不打包。
按准确服务身份启动和收尾；运行期间不进行安装、扫描、压缩、清理或远端配置
变更。与其他GPU实验不并行。不改与本项目无关的k3s。

先备份本登记及小型来源包，再启动一次测量。结束后确认真实GPU释放与空资源
域，停止仅本轮远端服务，保存双端传输证据，使用既有分析器
slora_resident_rr_v1/full4000_w1检查完整请求、原生计量、GPU积分与冻结SLO。
不补造未知数值正确性，不用不完整成本排名。新失败亦原样保留。

按冻结V1/单轮V2判断95%联合SLO、G1资源和G2预算；不改阈值，不计算运行间
CI。其他已完成W0/W1点不重跑。之后继续Loquetier/HydraServe W1，再返回
完整比较与有证据的PrimeLoRA优化；W2/G2、3B、消融、机制与敏感性均保留。
PrimeLoRA尚未达到跨场景目标，暂不画图，不把本次计量修复称为性能贡献。
