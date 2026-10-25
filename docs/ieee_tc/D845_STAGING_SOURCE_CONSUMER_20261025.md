# D845：准备前来源查询的独立调用点

## 已完成证据与单一假设

复用D843完整4000请求诊断的原始Python栈，无新采样、GPU运行或远端操作。
controller业务窗口1937样本、无缺失/截断；125个native回复解码样本逐条
按保存的调用行和当前AST归类，源码身份随清单保留：

| 调用点 | operation | 样本数 | 决定 |
|---|---|---:|---|
| GPU计划execute | source_snapshot | 61 | 保持完整；D514/D516/D519已否决此处identity替换 |
| prepare_gpu_source | source_snapshot | 41 | 本次唯一候选：来源准备前置检查使用新鲜identity |
| _plan_ieee_preparation_for_slot | routing_source_snapshot | 23 | 保持完整；规划实际消费全局物理图 |

频次不是CPU时间占比、独立重复或可删除时延。D843是有观察开销的诊断，
不能取代D840未插桩性能点。数值正确性未知和G1/G2未完成状态不变。

D691此前明确将prepare_gpu_source保留，因为当时591次source_snapshot
仅有operation聚合，不能归属实际调用点。此次41个直接调用链样本补齐了
这个缺口。它没有推翻D514的负结果，也不授权扩大到其他snapshot。

唯一假设：来源准备前置检查不消费物理分配图；以已存在的实时身份接口
读取同一owner的完整来源集合，可以减少该调用的物理图构造、传输和解析，
且不改变随后完整GPU执行观察、物理准入、引用或释放。闭环时序变化可能
抵消局部减少，完整同合同W2决定接受或撤回，不凭单测/采样宣布服务收益。

## 消费依赖与历史边界

prepare_gpu_source先等待本目标已有staging，再读当前来源：

- 已注册目标：核对adapter/path/content；冲突调用原复制来源失效见证。
- 原计划的native来源已消失：原完整身份/epoch/覆盖校验后结束旧目标。
- 目标尚未native就绪：保留原Remote→NVMe及native HOST准备链。
- 身份读取不持引用、不授权GPU操作；随后execute仍source_snapshot，
  ieee_prepare_host在原生事务中检查KV、E(t)、完整allocation/alias、引用和预算。

实际生产AST仅一处operation常量由source_snapshot变为source_identity_snapshot。
原生owner仍每次source_snapshot刷新，不跨调用缓存或TTL，不变更接收时间、
内容、epoch、选择、优先级、并发、profile、取消和失联规则。九式不变。
不新增RPC接口或fallback，也不删失效/错误路径。D474已有HOST入口优化保持。

D172/msgspec JSON已实现；D192逐项Python检查的新编码器已否决；D737规划
延迟解析经D738完整退化、D739撤回；D844事务内第二遍stat批量更慢，已关闭。
本轮不恢复上述方向，也不因为查阅序列化文档而实施另一并行codec候选。

## 原始依据与最小检查

本轮在线复核[vLLM 0.30序列化源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/serial_utils.py)
及[msgspec类型语义](https://raw.githubusercontent.com/jcrist/msgspec/main/docs/supported-types.rst)：
二进制表示与JSON有类型/边界差异，不据此再引入codec分支。
[Python asyncio](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
说明事件循环同步CPU工作会阻碍其他任务，但不证明本机收益或可节省比例。
本地gpu_monitor/residency_manager源码确认身份接口仍刷新同一原生owner，
完整GPU执行观察和提交边界保持。另在线核查[vLLM worker源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
原生add_adapter明确依赖单线程core，加载后才处理容量/激活；本候选没有将
源身份当作这些原生步骤的替代，没有借用外部性能数字。

复用原有mixed preparation、replacement、失效、取消、lost reply和identity
CPU测试；新增两个调用域测试覆盖远端/已有文件前置链，逐次对照真实owner
新鲜身份，并要求后续完整GPU观察及实际加载仍发生。保留D514的GPU执行
full-source断言。仅针对受影响路径，不重做广泛后端资格或新GPU短前缀。

分类脚本复用D843流式reader和窗口口径；所有输出独占新路径。
使用system-profile技能区分采样与因果效果；测试后按github-sync明确路径备份。

| 新增/修改 | 用途 |
|---|---|
| raw d845/classify_saved_rpc.py、curated源码快照 | 复用已有栈，125样本全部保留 |
| scripts/run_all_experiments.py | prepare_gpu_source唯一调用点；无新插桩 |
| tests/test_ieee_tc_host_execution_projection.py | 两条真实owner/完整后续观察回归 |
| 本文、EXECUTION_STATUS、curated JSON | 来源、负结果边界与候选状态 |

分类辅助域峰值140152832B，high/max/OOM/swap0；已正常退出。
67项针对性CPU检查最终通过（4.270秒测试体），完成域峰值567242752B，
high/max/OOM/swap0。生产候选在四次CPU检查间字节不变；没有GPU性能重复。
前三次失败全部保留：测试ContextVar被队列任务继承、测试将独立mutation
任务误当直接查询调用、默认handoff夹具被误用于residency。仅修测试归属
及复用夹具的原场景，不修改生产规则、伪造profile或删除失败。
一次工具连接失败发生在文档/资格入口写入之前，未启动测试或产生运行数据。

尚无新GPU性能成绩。备份后下一直接一次普通完整3B W2，沿用D840配置/profile/工件/输入/生成/
SLO/预算/真实交付，仅更换本次输出及缓存路径；不再profiling或重复组件计时。
最终7B同版本、其余3B比较、A1–A5、S1–S13及审稿文档/图表/V2完整保留。
