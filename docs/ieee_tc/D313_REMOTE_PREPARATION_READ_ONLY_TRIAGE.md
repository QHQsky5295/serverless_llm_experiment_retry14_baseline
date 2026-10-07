# D313：请求准备关联失败——运行期假设与终态证据

## 终态补充（2026-10-07 14:00+08）

D313已经结束，4000个请求均有终态，3999个返回成功、1个失败。失败为
`req_00216` / `support_lora_0040`，输入759token、目标输出7token；完整未截断
traceback确认异常来自`observed_preparation_interval`的317行。该请求的
`gpu_reference_evidence`确实为空，六项复合条件中究竟哪项失败仍不能由该行
恢复。下面的HOST/NVMe切换假设未被历史记录直接证实，不将其写成既定根因。

正常结束和真实GPU释放均已确认；生命周期观察值为15647.60522277304GPU-s，
不用于全正确排名。132次实际传输、131480060线上字节与远端记录对应，
无请求内打包。4036个资源样本中high/max/OOM/swap均为0。所有远端服务已按
捕获的身份核验并关闭，辅助域也已核验清空。原始结果保留，未运行新的GPU试验。

现有失败审计核对了全部输入，并对3999个成功返回检查原生身份、数量和时间；
这不构成数值正确性或正式SLO验收。结果位于
`paper_results/ieee_tc/p2_backend/d313_full_w0/`。`failure_audit1`保留初版清单
错误的历史（把正在写入的清单自身空文件摘要也列为输出）；仅离线修正并重新
发布`failure_audit2`，不修改原始实验、替换失败或改变指标。初版分析源码保存在
Git提交`b74cc96eb015fa70b92c0c993e3b63791c930221`。

下一步先用既有request-lifecycle fixture构造明确的合法文件层级变化，检查
是否可确定性重现同一校验缺陷，并保存最小失败判定证据。保持文件引用、内容、
owner、clock和边界检查；不以捕获忽略异常、禁用HOST或强迫慢路径来通过。
必须区分“可达缺陷已由测试证明”和“这个历史请求的具体根因已证明”。

## 以下为运行期间保留的只读定位记录

2026-10-07。这是待验证假设，不是根因结论或已实施修复。运行版本为
`0cf3c1dd10a4894516feff203f38d346ee5389aa`，D313 原始4000请求 W0仍在运行。
只读查看了少量源码、测试定义和 Git历史；未修改服务、执行测试、全量分析
运行结果或操作远端。原计划已完整复核，九个公式和冻结V1不变。

## 已观测事实

- `req_00216`，原生目标7token，终态时间856011.559573558，失败类型ValueError，
  `native_contract_matched=false`，终态无instance/prompt/token hash。
- source request SHA：`7ff6c2640bc25b95b5777e5123aa3bf30b735985bbdb1528e0d0f804ac726c15`。
- 服务日志的失败前缀对应`observed_preparation_interval`中的
  `remote preparation interval lacks its verified transfer identity`。
- 此复合条件同时检查adapter、published状态、时钟、内容验证、非空transfer ID，
  以及remote target path与native load path一致；仅凭异常名称不能判定具体哪一项。
- 13:06:50的最近服务进度为1119终态/1118成功/1失败。服务仍存活，资源样本1218
  无high/max/OOM/保护中止。本轮已不能满足全部请求成功的验收，失败不得删除。

## 当前源码的关联链

1. `scripts/run_all_experiments.py:_ieee_prepare_selected_adapter`（约17413行）
   对admission为remote的请求调用`_ensure_local_async`，然后取得native GPU lease，
   发布post-load身份并记录acquire，再计算准备区间。区间异常发生在生成之前；
   它仍是实际请求失败，不能当成可忽略的离线统计警告。
2. `_materialize_remote_adapter`（约18557行）在物理source owner管理下发布NVMe，
   保存确切transfer ID/target path/内容与时钟证据。
3. `_queue_ieee_file_preparation`（约19911行）区分IO creator和subscriber。
   `_materialize_remote_adapter_async`只把creator的remote证据交给该请求。
4. `_ensure_local_async`（约20264行）在远端完成后重新读取实际文件状态，
   在owner锁内按HOST优先、NVMe次之选择并pin当前有效副本。若请求拥有remote IO，
   它此前已经保存该次NVMe目标的remote准备证据。
5. `faaslora/preloading/preloading_planner.py:observed_preparation_interval`
   对remote/native两段要求相同文件路径；对native复用/共享文件准备则已有
   `profile_eligible=false,d_ms=None`语义，不能把未完整经历的路径计作零耗时样本。

## 可证伪假设（未确认）

请求自己完成Remote→NVMe后，后台可能已经发布同内容的HOST文件副本。
最终选取并pin的最快有效文件因此是HOST；remote证据仍正确指向NVMe，native
加载证据正确指向HOST，但上述直接路径等式不成立，导致请求失败。

该假设说明当前代码存在一条可能的触发链，不证明req_00216实际经过它。
也不能排除其余复合条件错误、交付失败或不正确关联。终态原始结果与远端日志
出来之前，不宣布“并发原因已证实”，不把它归因为带宽、OOM或D311的epoch缺陷。

## 历史与同类实现核对

- `6bdff25`引入actual preparation与service admission分离；对应测试检查真实
  起止边界、错误remote身份/路径和共享准备不成为零成本样本。
- `7297141`处理D90中creator/subscriber的区间归属；当前代码已有非空证据判断，
  不能再次把问题简化成空字典，也不能借用其他subscriber的时间。
- 已查看`tests/test_ieee_tc_preparation_cost.py`和request lifecycle中remote
  creator/subscriber、published HTTP路径测试。它们覆盖直接同路径与共享IO，
  本次查阅尚未发现“请求完成remote IO后选择另一已验证文件副本”的精确交错测试。
- vLLM0.30.0的LRU worker先验证加载，再驱逐/激活；其源码明确该方法依赖core
  engine单线程调用。这个边界不是Prime跨线程文件层级操作的同步保证，不能
  因后端串行就推定外层准备路径不可变化。
  [官方worker实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
- Python3.12的asyncio锁不负责OS线程同步。Prime的每adapter协程锁也不能单独
  证明后台文件发布停止；owner transaction仍需保留。
  [官方同步文档](https://docs.python.org/3.12/library/asyncio-sync.html)

## 本轮结束后的必要验证，不扩展实验矩阵

### 13:29 的只读补查：失败上下文的实际保留边界

当前源码显示，`_ieee_prepare_selected_adapter`在原生生成的内层`try`之前。
外层`_exec_request`仅为native identity/capacity两个特定错误复制reservation
上下文，随后按原规则清理reservation并重新抛出异常；本次preparation错误
不匹配这两个专用入口。最外层collector保留有限长度的Python traceback，
但此路径的`gpu_reference_evidence`默认为空。因此终态记录**可能只能确认
准确抛错位置，不能恢复复合条件中具体失败的字段**。

这由运行版本源码支持，不等于已经检查了尚未生成的D313完整结果。结束后仍先
检查真实失败记录和独立movement/transfer证据；若字段确实未保存，明确记作
不可恢复，不能将相近时间的另一份receipt拼成该请求的证据，也不反复扫描
相同日志试图恢复未记录的状态。现有失败分析入口可复用，但不得放松全成功
入口来容纳失败。后续若以确定性交错测试验证可达缺陷，必须区别“测试已证明
该缺陷存在”和“历史req_00216根因已证实”。必要诊断字段应在错误发生位置
保留最小身份/判定证据，不扩大线上快照、不重试或吞掉异常；当前尚未改代码。

1. 先正常收尾、确认真实GPU释放、核验远端三个InvocationID、停monitor优先，
   保存确切remote journal。大JSON使用现有流式投影，不整文件json.load。
2. 抽取req_00216的完整error、source admission、remote_preparation、native receipt、
   local source reference，以及同adapter的file movements。逐项检验上述六类条件。
   关联实际HTTP transfer和内容，不能只凭相近时间或adapter名称拼接。
3. 若确实是同内容不同已验证文件副本，使用现有fixture构造对应确定性交错，
   先复现再修正；否则否定该假设，沿失败的真实字段继续定位。
4. 修正须区分“可安全加载的已验证替代副本”和“该请求完整经历了所统计准备路径”。
   前者不应为保住旧统计假设而强制退回慢路径；后者不能通过重写路径/时间、
   填transfer ID、捕获忽略异常或取消内容校验来成立。检查是否可复用既有
   `confirmed_file_copy_pair`等身份验证，但其快照本身不是lease，不能冒充持有证明。
5. 保留错误owner/content/path/clock、非法stage顺序的负例，覆盖成功、取消和引用
   释放。真实服务D必须保留全部等待；不合格的完整准备样本不能更新d的EWMA。
6. 修正/测试/备份后再做必要验证与完整回放。当前没有新候选配置、GPU试验或修复。

仍按7B→3B→基线主线。正确性、共同warm/Resident标定和G1/G2尚未闭合，不画图，
不以条件延迟或CE代替全部正确完成。
