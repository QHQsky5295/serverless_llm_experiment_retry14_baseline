# D822：共享传输压力通道的完整3B W2检验

主线是D818提出、D819/D820实现验证的同一候选，不新增调参方向。
D817完整W2联合3596/4000=89.90%、U=11283.832798304502GPU-s，满足
既定75%资源预算但未满足95%联合SLO，且D711独立数值身份仍未知。
目标检验：用同一物理文件域权威共享压力替代每段IO对全部core的往返确认，
是否减少实际加载等待，改善共同SLO，而不是删掉压力项、缩短工作量或放宽阈值。

历史因果依据/官方源码/已否定方向见D818、D819、D820文档；不重复D595并行
RPC屏障或D628移hash锁的退化方案。IEEE九公式、分母和同core观测/使用语义
保持不变。D820的55项受影响检查及D819的12项真实跨进程检查复用，不重测。
D821实际原生48请求已完成；其原生读者身份/最终revision已核查，但来源测试
绕过Full admission witness，非零压力下的实际Full决策与扩缩容在本次直接检验。

## 单一变化与冻结输入

配置从D817复制，只有三个独占结果/缓存路径和resource_coordination下
ieee_file_pressure_transport=same_host_memfd_flock_v1变化。model/profile、
autoscaler、HOST/NVMe预算、min1/max3、输入和输出合同完全不变。
原seed42源trace4000条、既有W2 adapter map、458224输出token、真实远端
只读发布缓存继续使用。不生成工件或新负载、不重采profile、不执行短Full。
模型/调度配置不按W2优化；候选若有效还须同一版本W1/W0检验，不能拼版本。

既有D293 runner新增d822入口，PREFIX_COUNT为空；原生环境/descriptor、
64位时钟、watchdog及72/80GiB服务包络不变。预估新增峰值32GiB，启动重新
检查150/100GiB磁盘和实际内存，不依赖旧空闲量。一次一个重型任务，运行
期间不执行备份、压缩、构建、远端调整或第二次测试。

## 评价与收尾

V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22，
单轮V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d；
3B冻结SLO和D657 Resident参考不变，W2预算12083.06387024GPU-s。
先完整性、原生绑定/计量/实际释放，再联合SLO、U、P95TTFT/TPOT及等待分解。
检查Full真实witness中的generation/revision/active IDs、IO发布与owner退休；
publication不是per-core ACK，旧D594 RPC阶段分解不能用于该新通道。
不从历史时延减去估计开销构造结果，不将D711未知改为正确数4000。

运行一次并保留全部结果。结束先GPU释放和远端收尾，再按同合同分析/保存、
备份。若退化或错误保留失败并定位实际边界，不重复同键寻优、不加新阈值。
近目标前不画图。两模型同版主比较/Full、A1–A5、S1–S13、成本/CE/Pareto、
审稿文档和最终图表/V2仍全部保留。
