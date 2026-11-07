# D1117：当前 7B 实现的独立 G2 三副本工作点

## 已完成：2026-11-07 18:33

首次运行已结束并释放，4000原生合同通过，联合97.575%，6545.884585 GPU-s，
W1预算余量210.686465 GPU-s。完整结果和取舍见
`D1117_W1_RESULT_20261107.md`。原tmux已退出，不重启同键。
下一步为原登记要求的相同配置W2（D1118），不是另选参数或重跑W1。
下面RUNNING与NOT STARTED段落为历史状态。

## 当前执行：2026-11-07 17:54 已启动，尚未结束

登记/分析/配置备份提交 `13b94a6de06c625e58e22ac9dcc67b6c33733674` 已推送，
remote HEAD 核验一致。已按下述唯一命令启动，tmux `tc-d1117-7b-budget-w1`，
pane PID1300738；不要重启同键。服务scope为
`primelora-tc-svc-3865ea7edd1344129b8f54aa8327cbdb.scope`，辅助scope为
`primelora-tc-aux-315c2c79d9254d74a41e37f204ff3211.scope`。

本次拥有的7B远端invocation=`caf52e7ce72d44e39ef56fb2ad30306b`；启动时
两端均1Gbps/full。唯一初次health连接时服务尚在启动，随后共同delivery/timing
合同核验通过；此时尚未发部署通知，不是性能回放失败。外部通知monotonic
3552533.786908827，业务t0=3552593.786908827，仍提前60秒；source/view SHA
与下述W1登记一致。sample54无warning/abort，主机available114474233856B，
磁盘174124027904B；这是早期监控，不是最终性能或资源资格结论。

下一步只监控本次完整运行至真实终态/归还，之后收集远端对应clock的实际
journal（文件名不等于clock ID）、复用collector/review/W1 comparison。
原登记中的“未运行”语句描述启动前状态，不是再次启动的指令。

## 启动前冻结登记

2026-11-07。依据完整 PrimeLoRA-PLAN §P3/§M2 和 D1116 已存日志诊断。
这不是新的主协议、不是更换 resource_opt、不是重复 D1115；当前三场景
D1114/D1115/D1100 结果保留。下一执行先 W1，然后按相同配置检验 W2。

## 唯一因素与边界

`configs/ieee_tc/20261107_d1117_7b_budget_cap3.yaml` 的 SHA：
`34562e22c517fc62033da1c3437ef95df616fd54c8ae0dcb20a3e59db289eb53`。
相对 D1100 冻结 YAML，除新的结果/冷缓存路径，只改
`resource_coordination.max_instances: 4 → 3`。

保留 min1，四卡全部可见，14 slots/seqs、1024 batch tokens、FP16/TP1、
vLLM0.30.0、论文九公式、routing/handoff/hierarchy/admission、event-based
busy-service-rate forecast、38秒cooldown、45秒历史及全部在线阈值。
current-demand guard 仍关闭。不是强制3卡常驻，也不是按场景变更控制律。
retiring GPU 与新激活交叠可达四卡，真实持卡全部计费；max3不保证75%预算。

全体既有7B4000请求、500逻辑adapter、canonical prompt/固定输出、60秒
共同通知、真实远端已发布只读gzip合同不变。不新增数据、不提前使用未来
热点、不按phase清缓存、不等待实际drain后才发生下一段到达。

## 一个假设、两个场景

当前实现已消除若干旧请求路径问题，max4的W1联合98.55%但超预算847.30
GPU-s。D1116显示到达窗口四卡持有1061.57秒；这支持测试一个合法资源
工作点，不构成三卡性能预测。旧D619 W1可行、D621 W2不达标都保留。

1. W1 唯一新键 D1117：4000原生合同完整、共同joint≥3800、
   U≤6756.571050141065；报告原始P95、TPOT、E2E和全部副作用。
2. 若W1满足上述观测条件，下一唯一W2键使用同一配置，不再调参：
   joint≥3800、U≤12086.669343876012；相同D394阈值和D594参考。
3. 两场景同时满足才成为预算工作点候选；再以固定参考归一化P95等权比较，
   联合SLO差值单独报告。不将M1的资源与此点的延迟拼成单一优胜行。
4. W1若未满足，保留首次有效结果并拒绝其跨场景可行主张，不重复同键，
   不自动继续max2或改变阈值。根据具体瓶颈决定下一必要主线工作。
5. 全部数值正确性、基线适配和单轮局限保持原披露；不因已有未知项发起
   整套重复资格，不把时延与原生合同通过写成独立算术证明。

## 启动方法与验收

复用原 D1100 runner，仅新增可选显式配置路径；旧0/5参数行为不变。
在标准 GPU空闲、资源保护、磁盘/内存、远端身份/1Gbps链路检查及备份后，
以tmux运行一次（本登记时尚未运行）：

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1117_20261107/7b_budget_cap3_w1 \
  d1117_7b_budget_cap3_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1117_20261107/7b_nvme_budget_cap3_w1 \
  /dev/shm/tc-d1117-7b-budget-cap3-w1 \
  configs/ieee_tc/20261107_d1117_7b_budget_cap3.yaml
```

启动使用现有服务72/80GiB、swap2GiB，外置3/4GiB、swap0和固定CPU域；
推理机启动磁盘≥150GiB及增长余量、内存≥102GiB规则不变。只管理本次
拥有的远端invocation，不触碰共享k3s/其他服务。测量中不压缩、扫描或备份。
当前GPU/远端均空闲；启动前实际状态仍须重新确认，不沿用文档数值保证。

配置差异测试仅验证目标因素；旧设备域D618和当前后端资格复用，不再执行
无新问题的100/1000请求预跑。收尾先实际释放，再collector/review和现有
W1比较，保留所有失败及物理交叠，不改原D1115/D1100/D594数值。

此候选不修改生产算法；因此不需要重建后端或重新标定warm阈值。若比较
发现不利结果，按既定评价报告而不调整基线或消融以制造优势。
未完成的A1–A5/S1–S13、主表/Full统一、成本和两轮审稿文档不删除。
