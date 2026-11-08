# D1128 离线分析记录

全部分析在4000请求终态、GPU实际释放、服务域移除和远端inactive后执行。
各单元名称前缀为`primelora-tc-d1128-`，后缀为`-20261108.service`。

| 单元中段 | 工作 | 主进程 | wrapper | 说明 |
|---|---|---:|---:|---|
| metrics | 复用D1113完整计量 | 0 | 1 | 既有inotify ENOSPC，管理收尾timeout；实际域已移除 |
| review | 复用D299合同、传输与资源 | 0 | 1 | 结果成功保存，同类管理收尾timeout；实际域已移除 |
| stages | 复用D455阶段分解 | 0 | 1 | 主进程成功，管理收尾timeout；实际域已移除 |
| compare | 复用D1127/D1052 Full/CapacityOnly比较 | 0 | 0 | 4000输入和路由验证；192输出digest差异保留 |
| postrun-sources | 来源与旧文件保护核验 | 0 | 1 | 68来源/144旧文件/3用户dirty文件通过；管理收尾timeout，实际域已移除 |

所有分析CPU2/3/26/27、MemoryHigh3GiB/Max4GiB、swap0、CUDA隐藏。实际主
进程结束且域为空后才开始下一分析。没有重跑GPU或成功分析，也未修改全局
inotify限制。compare.py在真实释放后通过AST语法检查，再执行一次。

postrun-sources Invocation为fa47dfe720304d5791eb77640a0c17ff；wrapper最终
报告runtime111.007秒、CPU43.897秒、main0/timeout。14:57:37检查MainPID0、
ControlGroup为空。管理单元failed不等于已成功完成的GPU回放或统计断言失败。

远端journal在专属服务inactive/MainPID0后复制；132条均匹配本轮clock，
复制前后SHA相同。既有D299所需campaign-parent链接指向本轮原字节副本。
未改变远端配置/工件，未从请求结果事后扣除等待或传输耗时。

无独立语义评审；执行者确定性检查不冒称独立评审。保留未知独立数值
正确性、缺失准入回执及质量代价；不计算单轮CI，不按胜负选择保存结果。
