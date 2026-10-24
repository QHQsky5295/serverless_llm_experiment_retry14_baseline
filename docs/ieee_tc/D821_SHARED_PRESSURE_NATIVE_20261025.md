# D821：共享压力的实际原生进程验证

D820接入与55项受影响检查已备份。本次只验证D820在实际vLLM0.30进程边界
的可用性，不是Full性能点，不导出新的service/preparation profile。

复用D725原生来源测试的前六个完整波次，顺序为GPU warmup、Remote、NVMe、
文件HOST、原生HOST、GPU，共48个既有请求引用。模型配置仍是D724的3B图配置；
工件、prompt、target、生成合同均不新建。source spec仅声明新压力通道，
不从profile身份中豁免任何model字段。使用现有native_source_matrix、D298
启动入口、物理资源域、watchdog和真实远端已发布缓存；不新建实验框架。

核查实际reader订阅身份、IO开始/结束发布、native准入观测的transport/
generation/revision/活动区间、请求原生计数及实际退出。publication不冒称
per-core ACK；全部原始回执保留。现有控制来源设置是诊断条件，不冒充Full
自然命中率或95%主SLO结论，也不替代D711未知的独立数值身份验证。
同一worker来源测试不覆盖多副本自然扩缩容；D819真实跨进程和D820生命周期
检查继续复用，完整W2将验证实际扩容中的行为。

预测新增磁盘上界8GiB，采用既有推理机150/100GiB和内存72/80GiB服务包络；
启动重新实测，不沿用旧空闲量。远端部署和监控沿D817已确认的服务定义，
仅运行目录/monitor身份改变；不重新打包、不清全局cache、不改网络或共享k3s。

先保存启动身份和可回退代码，然后执行一次；结束后先核查物理GPU释放和
远端所属服务关闭，再统计并保存。通过后直接完整新键W2，不追加Full前缀、
不重采整套profile。不通过保留失败，依据具体错误处理，不能关闭压力检查
来获得成功。当前未启动时不宣称验证通过或性能改善。

## 实际结果（2026-10-25）

48/48完成且原生token数量匹配：GPU16、Remote8、NVMe8、文件HOST8、原生HOST8。
64段实际文件IO完成，均有同一generation的开始/结束publication；原生core末态
读取revision130、正确domain身份及0个在途。owner由attached实际转为retired，
domain关闭。48次诊断交付共111556980线上字节，双端SHA/字节一致，未按请求
打包。服务峰值5748473856B，high/max/OOM/swap均0；GPU真实释放、服务和aux
路径移除，三个所属远端服务关闭。原始数据与completion/summary.json留存。

必须限定：native_source_matrix的来源加载绕过Full admission witness，因而
本次证明真实reader连接/最终revision、并发publication及来源加载可用，
**没有直接观察到加载事务中的非零共享压力**，也未覆盖多副本自然扩缩容。
不把末态0伪作加载时0、不将其称为完整准入资格；复用D819/D820组件/集成
证据，下一直接D822完整W2，在实际Full witness验证该边界，不另做前缀。
既有数值正确性未知和G1/G2未完成保持不变，不导出/更换profile。
