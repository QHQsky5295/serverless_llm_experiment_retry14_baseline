# D879：受控层级初态与在线规划路径分离

同一D873/D874私有CPU构造候选，模型、工件、423条输入和已有case顺序不变。
D876因旧初始化身份不匹配在0到达时失败；D877/D878分别揭示collector把wave
编号冒作有效plan、随后又将空plan交给必须带plan的在线队列。全部失败保留。
本次不是绕过生产约束，而是纠正测量器职责：它显式清空自有初态并测分层
服务时间，不运行planner，所以直接调用已通过D875原生验证的CPU-only准备。

## 不变与改变

- 生产core/worker/residency/admission/Full代码及九公式不改；正常Full仍注册
  真正的计划，不能使用wave身份绕过校验。
- collector保留文件内容身份和引用，执行原生`prepare_file_host_and_hold`，
  核查owner/clock/lease/adapter，释放HOST引用后再释放文件引用。
- 容量不足直接令该次初始化失败，不在没有事件的队列中等待、不补零、不重试。
- 取消等待实际操作结算；RPC结局不确定时不声称释放，由失败关闭路径保留证据。
- CPU准备成本与时序全部记录；受控初态设置不冒充Full、handoff或planner性能。

依据是实际D878 traceback、D875原生资格和
[官方vLLM0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
的CPU构造与cache/activation边界；vLLM技能用于核对调用职责，不套用通用性能阈值。
4项针对测试实际调用native owner：HOST-only成功、容量拒绝、取消结算、未知
结果保留引用。首次测试fixture缺少真实worker所加clock字段的失败亦保留。

## 执行和出口

复用D701/D725 prepare/export、D298 source runner、D878远端管理；只新建
D879输出身份。同423cases、24个representatives、76waves，不新造trace/权重。
完整source.json、160服务类、准备类、原生输出、真实远端和物理释放由原严格
exporter检查。没有完成回执前不得称初始化完成。18项initializer及D874/D875
已有资格复用，不再另加短Full或共同SLO/Resident标定。

通过后，以新初始化绑定同一Full候选；先用已有ScenarioRunner加载器检查
model/config/context严格一致，再一次完整4000请求W2。95%联合要求、75%
预算及冻结参考不变。单轮开发观测，无CI；独立数值正确性仍未知。

raw：`results/ieee_tc/p2_backend_qualification/d879_20261026`；curated：
`paper_results/ieee_tc/p2_backend/d879_private_host_profile`。保留147旧投稿项、
用户manifest和全部失败。阶段性备份不等于最终V2同步。全部剩余实验与交付
继续开放，goal active，不因完成测量修复宣称PrimeLoRA性能达标。
