# D805：HydraServe 3B 原生交付数值失败与接收边界修正

本轮不是性能结果。保持用户确认的隔离部署，不接管属于其他项目的k3s；
继续复用原生控制器、provider、共享内存快速加载、GPU worker与HTTP入口。
原D804上下文错误已消失，但新交付检查发现具体数值错误，不能据此启动主矩阵。

## 实测结果与解释

| 检查 | 实测 | 可支持的结论 |
|---|---:|---|
| 原trace前100请求 | 100到达、100终态、100响应、0传输失败 | 请求/原生token数量合同完成，不等于数值正确 |
| 全零token输出 | 100/100 | 本轮模型交付不合格，不能作为baseline性能证据 |
| req_00001前三token | 实测0/0/0；D803原生/HF参考578/2326/1288 | prompt token IDs与adapter ID逐项相同，局部数值核查失败 |
| 实际adapter数/远程fetch | 29/29 | 复用既有500-ID池的100前缀；不是完整500覆盖 |
| 远端线上字节 | 67,350,156 | 真实已发布gzip交付，请求中打包0 |
| 服务内存峰值 | 30,710,648,832 B | high/max/OOM/swap均0，不是内存护栏中止 |
| GPU及远端收尾 | 已实际释放/停止 | 不留后台实验，不修改其他项目服务 |

原启动器`pass=true`只记录运行/资源合同，原始字节不追改。新增
`delivery_numerical_audit.json`明确`experiment_passed=false`、
`performance_result=false`、`n_correct=null`。独立参考只覆盖指定请求前缀，
不把它夸大成100或4000请求的独立数值证明。

## 根因证据

本轮共享池6GiB，即6,442,450,944B；序列化权重分配终点为6,425,520,819B，
余量16,930,125B，小于原生32MiB接收块。分配本身可容纳完整模型，但原生
`recv_into`始终请求完整chunk，末尾memoryview不足时产生`buffer too small`
异常。两个实际加载进程均在`local_server.py:210`触发同样异常。

原生父进程等待子进程不再存活后，未核验exitcode便把进度标为完整；实际
worker日志随后确实看到`cur_recv_end = 6425520819`并继续初始化。provider
另记录对端重置连接。这构成具体的接收失败与错误完成发布链，不是调度慢、
机器规模小或LoRA远程传输慢的推断。哪些末尾张量导致全零尚未单独逐张量
验证；不能提前声称修正后GPU输出已恢复。

已联网复核[HydraServe原生接收实现](https://raw.githubusercontent.com/LLMServe/HydraServe/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/kubernetes/vllm/storage_server/local_server.py)
及[Python socket接口说明](https://docs.python.org/3/library/socket.html#socket.socket.recv_into)。
旧7B成功并不排除此边界错误；它取决于实际映射末尾余量，不能靠为3B额外
增加任意内存或缩小人工chunk掩盖。检查既有修改记录，未找到已经实施的
同一接收边界修复。

## 必要修正与验证范围

沿用现有`shm_control.py`和`native_delivery.py`，增加显式
`bounded_shm_receive`开关：每次接收长度取`min(chunk, end-current)`；
提前EOF失败，父进程join并核验所有接收进程成功后才发布完整进度。
不变更原生双进程、分区、线协议、渐进可见性、启动重叠、调度/批量、
模型数学、LoRA内核、GPU容量或SLO。旧默认overlay字节保持原样，D805的
实际执行快照保持修正前身份。修正是边界正确性，不是baseline性能调优。

37项CPU检查通过（1.607s，工具session31785，exit0）；其中真实socket小
buffer重现原始异常，修复版本覆盖精确尾块、分段到达、不越邻接分区、
提前EOF、失败子进程不发布完成及成功发布。其余原生输入/模型/控制测试
保留。CPU检查不替代共享权重GPU数值确认。

使用run-experiment/monitor-experiment维持受控单次运行，analyze-results
将响应合同与数值正确性分开；serving-llms-vllm的加载诊断路径辅助定位，
不采用其示例性能阈值或调参建议。github-sync用于失败与修正的显式路径备份。

## 来源、收尾与下一主线

- 原始目录：`results/ieee_tc/hydraserve_qualification/d805_20261024/`（主实验仓库）。
- 实际BASE执行提交：`71b2ba170ddbd9b753e689c20f795f542d129cec`；
  MAIN为`6e41733b78b1b7533a0f858f4e691a2845552a6c`。
- launch SHA：`69f357671175d53957ad24fc9ed8cc64aa93af2fc1d7b16ba8c1075ca0609ff8`。
- source manifest SHA：`8a5a284107754c51ba789cc795ab7f21fc5daff742838561ae7d152172dc27c4`。
- 数值失败审计SHA：`c15732626f94c0b75e210152a1b429b626120f0ea3862020405ee44852304755`。

继承的collector错误查询7B传输日志，但精确InvocationID停止步骤已成功。
保留该错误，以只读复查实际3B clock_id、拷贝与双边SHA核对补齐日志；没有
再次启动/停止服务。audit域已关闭，峰值127,410,176B、保护事件/swap0。
下一新的检查入口应显式按实际model采集，不再沿用错误日志路径。

先备份本次失败和必要修正；回收经保留副本/引用/SHA核验的本轮重复缓存，
恢复150GiB启动余量；再在新执行身份验证受影响的交付路径，不重做平台与
未变模型数学资格。通过后完成HydraServe3B W0/W1/W2，再完整比较及优化
PrimeLoRA。同版Full、A1–A5/S1–S13、成本/CE/Pareto、审稿文档、图表和
最终V2继续全部保留。此检查不是整个目标完成。
