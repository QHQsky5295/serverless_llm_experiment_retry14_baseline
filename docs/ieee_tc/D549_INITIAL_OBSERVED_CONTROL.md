# D549：初始副本等待期间的真实到达控制

2026-10-15。D548 的单一假设实现与 CPU 资格检查；不是新的 GPU 性能结果。
完整 PLAN、冻结指标 V1／单轮 V2、最新 baseline 范围及安全规则继续有效。
不改论文、公式、既有运行或 baseline。候选默认关闭，旧配置行为保留。

## 1. 为什么只改这一处

D533 W1 的首个引擎在业务开始后42.645秒就绪，控制事件从42.910秒才开始。
源码中 preload 等待首次 ready，随后 run 才启动周期控制；外置入口已经记录
真实请求到达。D548 显示早期接纳等待与实际副本启动重叠，但没有证明所有
违约都由此引起：251条联合违约中218条在initial ready后到达。

本次可证伪假设：让初始等待期间已经观察到的队列也进入原有 IEEE 控制器，
可能更早提供容量并减少后续积压。更早持卡也可能增加总GPU-s；只有完整
新版本回放才能决定是否接受，CPU检查不能证明性能收益或95%达标。

历史依据是 D466（接纳容量）、D530（阶段与资源）、D533（W1）、D548（违约
与控制时序）。不重新做已修正的FIFO许可唤醒、CPU-only frontend、原生异步
scheduler，也不重启D529已拒绝的元数据候选或D386激进queue阈值扫描。

## 2. 论文规范语义与本次实现

| 论文规范语义 | 实现与可核验边界 |
|---|---|
| §3.1每控制周期计算三项归一化压力的最大值 | 直接调用同一个`_evaluate_ieee_live_control`及`IEEEReplicaControl`，不另写一套启动控制公式 |
| 排队需求形成扩容压力 | 只取外置入口`observed_times`的实际到达数，不消费请求，不读取未来trace填队列 |
| pending不是ready，副本总数受限 | 初始和自然扩容统一计入pending；设备预留和现有物理owner仍约束启动 |
| 未知TTFT不代表低时延 | 初始阶段无完成样本，TTFT保持未知，不因此允许缩容 |
| 新副本必须真的ready才服务 | 可由最先发布ready的initial或natural副本服务，类别不互相改写 |
| 生命周期与失败不可隐藏 | 启动任务在创建时即登记；取消须等原生启动收尾，再关闭共同搬运owner；失败不重试、不选成功兄弟掩盖 |

接口为`resource_coordination.ieee_initial_control_policy: observed_arrivals_v1`。
未知值或缺少外置入口直接拒绝；未设置的历史配置保持原路径。该开关标识
待验证的新执行版本，不是看到某个场景成绩后自动切换策略。

没有改变九个行间公式，也没有修改`autoscaler.py`。仍是每周期最多新建一个
副本，使用原有上下限、TTFT窗口和cooldown。没有固定额外常驻副本、未来
热点、GPU预热、增加并发上限或时间扣除。相同模型的新版本须跨W0/W1/W2
采用同一规则；当前还没有发布新的性能配置。

初始阶段产生的scale-out事件与后续回放共用同一个结果对象，不在run入口
丢弃。控制器对象和上次采样时间也保留。`first_ready_activation_kind`记录
首次服务来自initial还是natural；`ready_events`仍只记录initial成员。
所有实际GPU allocation/release继续来自既有物理计量，不用逻辑ready数替代。

## 3. 官方资料核查及不能迁移的结论

- [Llumnix，OSDI 2024，§4.3–4.4和§6.5](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)
  将在线负载与实例扩缩容协调起来，并同时报告尾延迟和实例资源。它使用
  freeness与KV迁移；本次没有移植这些算法，也不以该论文证明提前控制必然有效。
- [vLLM 0.30.0 scheduler源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/scheduler.py)
  的async scheduling针对GPU执行间隙；本机早已启用，不能把它重复算作本次改进。
- [Python 3.12 asyncio任务文档](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.wait)
  明确`wait`的timeout不取消pending任务。因此控制采样等待不会周期性取消
  初始化；真正取消仍走现有拥有者的shield/join收尾，而不是抛弃后台任务。

以上于2026-10-15重新读取。实施依据仍是本机保留日志和IEEE语义；不扩展成
无关综述、不把标准异步编排包装成论文的新算法贡献。

## 4. 资格与失败记录

复用`tests/test_ieee_tc_launch.py`中的真实控制、初始生命周期和激活路径，
后端为CPU合成fixture，不申请真实GPU。新增检查覆盖：

- 有4000个未来trace条目但零真实到达时，不额外扩容；
- 已观察队列越过原阈值，可在initial pending时启动natural副本；
- pending总数达到上限后不再创建，未知TTFT不造样本；
- natural先ready时立即成为可服务主副本，initial身份仍保留；
- 早期scale-out结果与控制器连续，不丢记录、不重复创建最小池；
- initial或natural失败均不能被成功兄弟掩盖；
- 取消等待两类原生启动收尾之后才关闭共同搬运owner；
- 完整handoff路径中两类activation的准备工作都保留并可正常结束。

第一轮31项，30通过、1项fixture失败：NoHandoff没有启用搬运队列，却要求
关闭回调执行一次。失败日志和当时源码压缩归档保留。修正为显式绑定空队列，
并新增真实Full准备路径测试；生产代码没有因这个失败而改动。
第二轮32项全部通过（49.648秒），未发生high/max/OOM/kill/swap事件。
两个辅助域均已核实递归成员为空、关闭且路径消失。
完整受影响模块回归363项全部通过（74.895秒），包括初始/常规控制、CPU
frontend和共享搬运；峰值2488819712字节，high/max/OOM/kill/swap均0。
回归域也已核实递归成员为空、关闭且路径消失。没有真实GPU性能结果。

原始证据：`results/ieee_tc/p2_backend_qualification/d549_20261015/`，位于
original实验项目的既有results目录。使用独立D549键，不覆盖旧结果。
CPU辅助域3/4GiB、swap0、CPU2,3,26,27；首次/第二次峰值分别
1960259584/2250047488字节。使用get-available-resources只读核查资源，
run-experiment复用原测试入口；不新建实验框架、数据或工件池。

## 5. 后续验收，不提前接受候选

先保存经回归检查的代码节点，再按既有100请求资格路径核查真实外置到达、
原生启动/释放、生成数量、tier快照、早期自然控制事件和无资源泄漏。资格
通过后才运行新键的完整7B回放，使用原有输入、LoRA、真实远端发布对象、
生成契约、共同SLO和GPU-s口径。若资格负载未触发早期控制，不能声称验证了
该机制；补充有明确身份的功能检查，不伪造触发或更改正式到达。

全量结果同时看联合SLO、P95/TPOT、U及阶段/激活计量，不从旧TTFT减去
启动时间构造成绩，不重复同键寻找赢家。先判断W1假设，再核查W0/W2同配置
副作用；当前93.725%的历史W1点不因代码改变而升级为达标。

仍待完成的完整范围：7B W1/W2/G2及共同参考、3B优化与所有既定baseline、
dLoRA B6、A1–A5、S1–S13、离线成本/CE/Pareto、两轮审稿设计文档、最终
学术图表与V2交付。现有HydraServe隔离适配已完成，不碰其他项目k3s。
