# D734：当前 3B 请求接收等待的有界函数观察

## 问题与停止边界

D733 已复用 D732 全部请求，发现文件观察只与累计提交→接收等待的
18.989% 重叠；182 条 TTFT 违约中仅23条在这段等待中与文件观察重叠。
这不是因果扣减，但足以说明尚不能直接将整个文件 owner 异步化当作首选。
本次只问：当前3B controller中，哪些同步调用与接收/返回等待并存？

复用原1000请求前缀一次，不生成负载、不改来源/到达/生成/工件。不增加
Full性能重复，不调控制参数，不改九个公式，不重做原生HOST profiling。
历史D456/D603是旧7B；D684只有native spans，不能替代当前controller观察。
若已有采样分辨率仍不能定位，报告限制，不无限追加局部诊断。

## 方法与解释限制

- 复用 `profile_d328_controller_python.sh` 和既有 `python_frames_v1`；
  每2秒记录Python线程frame与现有source RPC边界，不新增插桩。
- 保留当前D732全部系统配置，只换输出、NVMe工作缓存和HOST缓存的独占路径。
- 禁用额外native core/step/HOST清点观察；外置原到达回放和安全监控不变。
- 采样受GIL和调度偏差影响；frame频次不是无偏CPU时间，也不证明独占因果。
  记录实际采样间隔和observer开销，不从请求时延中扣除。
- 1000前缀不是4000主结果；带观察运行不参与G1/G2排名，也不与D732宣称性能胜负。
  独立adapter数值正确性未知仍为未知，不由token计数或HTTP成功推定。

本次复核[Python asyncio官方线程边界](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
与[vLLM官方CPU路径分析](https://vllm.ai/blog/2024-09-05-perf-update)，只采用
按实际组件区分同步工作、异步等待和GIL竞争的方法，不移用其加速比。

## 启动、保存与恢复

使用既有run-experiment入口及tmux；服务72/80GiB、swap2GiB、CPU4–23/28–47；
外置回放/监控3/4GiB、swap0、CPU2/3/26/27。既有磁盘150/100GiB和内存
余量公式不变。每次只有一个重型任务，不在运行中备份、扫描、压缩或改远端。
真实远端复用D80已发布只读gzip；请求不打包，不复制工件。保留真实传输。

运行结束沿用实际GPU释放后按InvocationID关闭本轮远端服务；旧数据不覆盖。
配置、启动脚本快照、检查结果与来源放入
`paper_results/ieee_tc/p2_backend/d734_3b_controller_prefix1000/prelaunch/`；原始日志
在 `results/ieee_tc/p2_backend_qualification/d734_20261022/`。
启动前显式路径提交备份，V2分歧不force，不stage用户manifest。

## profiling变更清单

| 文件 | 本次改动 |
|---|---|
| scripts/ieee_tc/profile_d328_controller_python.sh | 只登记D734使用已有stack observer |
| scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh | 登记1000前缀、3B输入/远端、已有observer入口 |
| configs/ieee_tc/20261022_d734_3b_controller_prefix1000.yaml | D732配置，仅三个独占工作路径变化 |
| tests/test_ieee_tc_d456_diagnostic_contract.py | 两项针对性合同检查 |
| raw/d734_20261022/ | 复用原启动/关闭/health脚本；保存检查与运行回执 |
| 本文与EXECUTION_STATUS | 问题、边界、状态与下一步 |

没有修改生产推理/路由/规划代码。该观察仅由D734诊断入口启用，后续普通
性能入口不启用；保留脚本作来源追溯，无须删除历史证据来“清理插桩”。

## 共享集群与完整主线

用户已确认旧k3s不是本项目；已有HydraServe独立rootless Podman适配及7B
W0/W1/W2证据继续复用，不访问旧集群、不索取kubeconfig、不重做基础适配。
3B优化尚未达到95%，后续W1/W2、baseline/dLoRA、7B最终版本一致性、
A1–A5、S1–S13、CE/审稿文档和最终图表全部保留；当前不画图。
