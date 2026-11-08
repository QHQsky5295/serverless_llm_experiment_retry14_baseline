# D1127 离线分析记录

全部分析在4000请求终态、GPU实际释放、服务域移除和远端inactive后执行。
各单元名称前缀为`primelora-tc-d1127-`，后缀为`-20261108.service`。

| 单元中段 | 工作 | 主进程 | wrapper | 说明 |
|---|---|---:|---:|---|
| metrics | 复用D1113完整计量 | 0 | 1 | 既有inotify ENOSPC，stop-sigterm timeout；实际域已移除 |
| review | 复用D299合同、传输与资源 | 0 | 1 | 主进程成功，实际域已移除，同类管理收尾timeout |
| stages | 复用D455阶段分解 | 0 | 0 | 正常完成，实际inactive/dead |
| compare | 复用D1126/D1063 Full/NoHOST比较 | 0 | 0 | 4000输入匹配，191输出digest差异保留；实际inactive/dead |
| postrun-sources | 来源与旧文件保护核验 | 0 | 1 | 69来源/144旧文件/用户manifest通过；管理收尾timeout，实际域已移除 |

所有分析CPU2/3/26/27、MemoryHigh3GiB/Max4GiB、swap0、CUDA隐藏。实际主
进程结束且域为空后才开始下一分析；没有重跑GPU或成功分析，没有修改全局
inotify限制。compare.py在真实释放后通过AST语法检查，再执行一次。

远端journal在专属服务inactive/MainPID0后定位；内容clock与本轮health
相同，复制前后SHA相同，132行。既有D299所需campaign-parent链接指向本轮
原字节副本。未改变远端配置/工件或将打包耗时从请求结果中事后扣除。

没有独立语义评审；执行者确定性核验不冒称独立审稿。保留源结果、未知
独立数值正确性及所有质量代价；不计算单轮CI、不按胜负选择是否保存。
