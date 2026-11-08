# D1126 离线分析记录

所有性能分析在4000终态、服务域移除、GPU为空及专属远端inactive后执行。
复用D1113/D299/D455、D1124比较及D1054时序方法；未重跑GPU、未改历史结果。

| 单元 | 工作 | 主进程 | wrapper | 说明 |
|---|---|---:|---:|---|
| run-u395.service | D1113完整计量 | 0 | 1 | stop-sigterm timeout；实际域为空后移除，inotify ENOSPC |
| run-u396.service | D299合同/远端/资源 | 0 | 1 | 同类收尾timeout；实际域为空后移除 |
| run-u397.service | D455阶段分解 | 0 | 0 | 正常完成 |
| run-u398.service | 当前7B Full/Delayed对照 | 0 | 0 | 4000输入匹配，191输出digest差异保留 |
| run-u399.service | 实际handoff时序 | 0 | 0 | 0已观测违例，252缺invoked标志的deferred回执列未知 |
| run-u400.service | 来源及旧文件保护核验 | 0 | 0 | 65唯一来源、144旧投稿文件、用户manifest通过 |

上述均为3/4GiB、swap0、CPU2/3/26/27、隐藏CUDA的离线域。已验证实际主进程
终态和域释放后才启动下一分析，不把wrapper等待当成继续计算或重新运行理由。
compare.py及audit_delayed_timing.py在测量结束后通过AST语法检查再执行。
时序脚本在看本轮结果前明确区分已观测检查与证据完整性，未知不作为性能
准入失败。未做独立语义评审，执行者确定性核验不冒称独立审稿。

远端未安装rg，转用find定位原日志。初次以clock UUID猜测文件名未找到，
无数据复制；随后从实际journal内容核对clock，复制原始字节且双边SHA一致。
没有更改服务配置、工件或任何性能结果。
