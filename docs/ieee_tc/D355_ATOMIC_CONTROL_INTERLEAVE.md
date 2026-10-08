# D355：完整控制事务与原生模型推进的有界交错

2026-10-08。主线仍是7B，不恢复baseline或改变共同尺子。

## 假设与选择依据

D353完整4000请求的暂定联合时延通过3752/4000，未到3800；短输出
TPOT超界145个。D354复用D351诊断发现：8533个非空且进入时有未完成
请求的输入轮次中，2085个含多个UTILITY；108个短输出超界请求中104个
与同轮后续控制操作相交。该证据是有观察开销、pre-D352的相关性，
不是可直接扣除的延迟，也不证明GPU空闲。详细数据见D354文档。

已核对D333原子pending/GPU保护、D329跨请求HOST缓存反例、D352同事务
HOST证据，以及[vLLM 0.30.0官方core](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)。
原生循环在输入处理结束后才执行模型步骤；输入处理会排空可用队列。
候选只改变此处公平性，不删除物理清点或重做不安全的跨请求缓存。
[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
提供检查不同工作的执行干扰这一方向，但其prefill策略不等于本次CPU
控制事务优化，论文性能数字不转用于3090或Prime。

## 唯一变化

`model.ieee_control_interleave=atomic_control_yield_v1`，默认缺省不启用。
每处理完一个Prime控制调用，若native `is_running()`且`has_work()`，
返回原生busy loop，让它执行原本的模型步骤和计数发布，再继续FIFO。
没有人工毫秒阈值、额外线程、递归模型步骤、输入重排或新增缓存。
一次完整事务是自然执行单元，不扫描正式结果选择控制时长阈值。

Prime控制调用由已有API身份判定：七个`ieee_*`控制入口及
`collective_rpc(ieee_gpu_reference, ...)`；不是按模型名、数据点、
结果胜负或观测到未来请求分类。其他原生请求仍由原生handler处理。
同步返回后才交错；Future返回的retirement仍由原生callback完成，
不伪造已退休。调用失败的原生错误返回也不改成成功。

保持不变：

- IEEE九式、routing/planner/admission候选、实际KV和存储检查；
- D333 pending注册与GPU引用保护、HOST准备及admission的事务原子性；
- 原生模型step、异步batch、abort旁路队列、shutdown和回包身份；
- D353容量14、软目标8、profile、生成合同、W0输入和发布制品协议；
- G1/G2及D170暂定开发尺子，不把完成数量当N_correct。

只支持已限定的dense TP/PP/DP=1、精确UniProc执行器和单core/thread。
安装时检查vLLM版本与原生输入函数AST；未知实现拒绝运行，不静默回退。
规范化Python3.12新增的空`type_params`字段，不忽略可执行AST变化。
控制操作可以在独立事务之间看到模型推进后的新状态，原有重检查仍执行。

可预期代价：控制回包、admission和准备可能变慢，batching与扩缩容也会
变化。改善TPOT不自动意味着联合SLO或GPU-s改善，必须完整复测。
新增累计计数仅说明输入轮次、已处理调用及让出次数；不宣称GPU真的
执行了多少kernel，或每次让出必定有token产生。

## 已完成资格（非GPU性能实验）

- CPU环境110项测试通过；安装vLLM的环境同110项通过。
- 实际安装的EngineCoreProc原生handler/step接口，在合成executor与
  工作上通过FIFO、回包和交错检查；未初始化CUDA。这不是实际推理资格。
- 覆盖idle阻塞/空队列、ADD/ABORT、shutdown、异步retirement、错误、
  source/version/topology/owner拒绝及真实D333事务方法。
- 首次109项测试失败保留：AST哈希受Python3.10/3.12空字段差异影响，
  以及非法开关在缺scheduler时走到旧通用错误路径。修正空字段规范化，
  且对显式native交错配置一律失败即停、不重试其他配置；不是放松保护。
- 首次源文件归档、日志、后续通过日志及源SHA均保留。没有GPU重试。

## 下一验证及验收

复用既有runner、真实远端发布缓存、原W0前100请求；唯一功能变化为
上述开关，输出/HOST/NVMe路径隔离。该前缀用于正确性与触发，不能外推
完整GPU-s或宣称正式SLO合格，不强制100前缀一定显示性能优势。

验收：100/100终态与原生固定输出、请求/adapter绑定完整；无fallback；
新增计数实际出现且符合单调/子集关系；GPU真实释放、预算/资源护栏
无违规，远端请求内打包为0。核对TTFT/TPOT分解和输出变化，并保留全部
失败。通过后才以相同固定配置做普通Full4000，对照D353及历史D336，
同时报告联合通过、TTFT/TPOT、GPU-s，不筛赢家、不叠加参数修改。

原始目录`results/ieee_tc/p2_backend_qualification/d355_20261008/`，
汇总`paper_results/ieee_tc/p2_backend/d355_control_interleave/`。
Plan/V1 SHA沿用D354；最终warm/Resident/N_correct及3B、基线、消融均OPEN。
