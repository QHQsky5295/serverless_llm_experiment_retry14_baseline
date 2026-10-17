# D614：下一项资源工作点的依据与验证边界

2026-10-17。本次仅核对既有证据、论文、历史源码和官方资料，没有新GPU实验，
没有修改生产算法或工作点配置。延续D611质量达95%但超预算的优化主线。

## 观察与历史反例

- D611完整W2联合95.475%，U15159.982437，预算12086.669344 GPU-s。
  D612确认最后三个源序时间窗持续四卡，新增76.610597到达窗GPU-s均在前段。
- 后半程控制采样的active/ready容量均值约0.0898、0.0716，不能等同于GPU
  utilization，也不能由此证明少一张卡不影响排队。准备分段仍主导多数违约，
  前端/控制等待随工件集合增长；因此需要验证资源工作点，不能只继续追局部
  文件观察的微小加速并默认它会降低生命周期U。
- D386将queue_upper从8改4，完整W0物理U增加23.50%，不采用更激进触发作为
  无证据补救。D380的8/14容量比较同时换了profile/开关，不能视为单因素。
- D595/D598并行压力读取候选的完整W2退化已回滚（D601），不重新叠加。
  D606/D610的fresh-file观察候选及其完整D607/D611结果保留，不删除新鲜度、
  身份/容量校验或安全边界。当前生产residency_manager SHA仍为
  fd783131934664f1dee49f78b21366c2aff7264d60a15c2a9bcb975e207bec4f。
- D593完整六系统7B W0/W1/W2比较和D594共同参考均复用；不重做基线适配。
  现有configs/ieee_tc中精确`max_instances: 3`检索无命中，这仅表示该配置
  目录未见已登记三副本点，不声称整个机器历史中从未出现任何三卡运行。

## 论文与实现约束

IEEE附件SHA35da2d89941452df2959b022aedddd4a984e45b8c15efb9951b5295deb198b90，
设计§3.1规定三项归一化max-score、configured replica limit、all-low持续
cooldown、无pending才回收。`IEEEReplicaControl`在faf5c85接通该语义，
ready+pending计上限，pending不冒充ready容量。P1-D48和当前源码一致。

因此不删除TTFT条件、不改变饱和度分母、不把正在启动副本当作可服务容量、
不改cooldown/窗口/upper/lower来追随本轮结果。min/max是论文已允许的部署
工作点，不是第十个优化公式。PLAN §P3.3/§M1/M2允许模型特定配置且跨场景
固定，§S5明确主配置选择仍允许3卡；共同四卡是上限，不要求一直使用四卡。

## 原始资料核查（2026-10-17）

- [Llumnix OSDI 2024 §6.5](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)
  同时展示尾延迟与平均实例数，并通过不同扩缩容设置比较质量—成本关系。
  这支持验证不同资源工作点，不证明PrimeLoRA三副本可行，也不移植其KV迁移。
- [ServerlessLLM官方历史控制源码](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)
  auto_scaler对请求驱动的desired_instances应用min/max边界；副本上限是部署
  约束，不等于固定常驻实例数。本次不改其已经封存的基线配置。
- [vLLM0.30.0优化文档](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/)
  区分批量/token预算对TTFT、decode和内存的取舍；
  [固定版本scheduler配置](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/scheduler.py)
  提供并发序列与token边界。现有证据不支持同时更改batch、slot、token预算。

## 一个可证伪的下一候选

在D611已测试实现上，仅将7B模型级`max_instances`由4设3；min=1、TP=1，
仍按现有在线max-score及回收规律在1–3个管理副本间变化。保持14个原生槽、
1024 token预算、全部profile、层级/admission/路由、CPU/HOST包络和真实远端。
仍有相同四卡可用环境，不把物理GPU清单改成不同机器，也不占用空出的GPU
做另一实验。该配置不按W0/W1/W2、run或未来请求切换。

假设：既有准备路径的效率改善与较低观测活跃容量，可能允许更少副本仍达到
共同SLO，且减少持卡和多副本控制工作；反例是排队/加载争用增加、联合达成
低于95%或排空过长导致U仍超预算。三副本上限不保证75%生命周期预算：
准备、drain、清理及退役/新激活的物理重叠均须真实计量，不截断、不换分母。
不能拿三卡×trace长度作结果，也不把手工配置称为学习得到的自动控制律。

下一步：复用既有launch/control测试验证该上限与pending/容量/清理一致，
新配置逐字段只允许max_instances及唯一输出/cache路径差异；冻结、备份后
优先完整W2验证唯一新执行键。若W2失败，保留失败并诊断，不重跑同键；
若通过则以相同配置完成W1/W0，不能拼接旧四副本结果当新resource_opt。
不增加100请求重复作为性能证明；需要的功能检查使用既有CPU合同和已有原生
资格，若出现具体集成缺口才补最小原生检查。尚未创建/启动该GPU执行键。

本候选是已授权的模型级资源工作点验证，不是新算法贡献、单机制消融或
最终赢家。独立数值oracle未知仍保留。完整3B/dLoRA/A1–A5/S1–S13/成本/
审稿文档/最终图表继续开放；所有旧候选及退化保留，当前不画图。
