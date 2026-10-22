# D723：当前3B前100请求的模型调用组件诊断

## 预登记问题与边界

D722复用D684原始数据：974个含多token分配的step平均wall64.547ms、
线程CPU64.214ms，scheduler仅0.305ms。D721完整请求重叠只能显示关联，
不能证明prefill计算或token预算是瓶颈。当前D719完整联合SLO91.60%，
U13508.300779GPU-s，独立数值正确性未知；本诊断不改变这些事实。

只运行既有3B W0源trace的前100条，复用既有500池、D701 profile、D719
Full配置和真实已发布gzip服务。配置仅改输出/NVMe/HOST隔离路径，token
预算4096、SLO、路由、admission、handoff及九公式不变。一次、开发性、
包含观察开销，不作为M1/M2性能点，不把前100当完整4000。

使用已有host_spans_v1和v2_components_v1，观察execute_model、输入/
attention准备、LoRA激活、graph主机调用、sample/output等。无新GPU同步、
CUDA event或逐tensor探针。沿用64MiB/200000span每进程硬限，预先缩短
观察窗口而非重复D519已出现截断的1000请求配置。若仍截断，保留限制，
不自动扩大日志限额或重复至完整。不开HOST inventory附加观察。

假设：多token分配step的主要host成本可定位到至少一个现有组件；若主要
处于execute_model未覆盖段或异步等待，明确未定位，不直接降低token预算。
host wall/thread CPU不等于GPU kernel时间；嵌套组件不重复相加，异步step
可能回收前一batch，分配形状不是纯prefill/decode标签。

历史D529的LoRA元数据改动未改善完整W0，不因组件再次较慢直接恢复它；
D687/D688 HOST元数据、GC/同owner串行/scale-down等拒绝证据仍有效。
后续只有新增因果依据才提出一个语义保持候选，不能堆叠补丁或扫参数。

## 官方资料核查

[vLLM v0.30.0 model_runner](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/model_runner.py)
的execute_model包含请求状态更新、输入准备与实际模型调用，不能把总时间
都叫GPU计算。[官方profiling说明](https://docs.vllm.ai/en/v0.30.0/contributing/profiling/)
明确观察功能会影响性能，建议有界请求量；因此本轮只作诊断。

## 观察功能变更清单

|文件|变更|生产语义|
|---|---|---|
|scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh|注册D723前100及3B/observer入口|历史分支字节还原核查|
|scripts/ieee_tc/profile_d328_controller_python.sh|D723启用既有组件观察|默认与旧campaign不变|
|configs/ieee_tc/20261022_d723_3b_step_components100.yaml|D719三个隔离路径|无策略参数变化|
|tests/test_ieee_tc_d456_diagnostic_contract.py|精确双白名单及D723路径/前缀测试|未放宽默认观察边界|
|原始D723目录管理脚本|复用D718远端管理及D684资格检查|不改远端冻结服务|

生产core_timing.py、runner及analyzer不改。system-profile负责有界观察，
run-experiment/monitor-experiment沿用既有资源门槛和真实进程检查。
服务72/80GiB、辅助3/4GiB及实际CPU亲和性、内存/磁盘门槛不变；不触碰
共享k3s。旧投稿144文件和用户manifest核对；原始大数据不入Git。

69项CPU检查通过，峰值696946688B，high/max/OOM/swap0；144旧投稿和用户
manifest SHA不变。首次辅助scope名称不符合既有UUID规则，被启动核验拒绝；
修正名称后，旧D519静态测试因新增白名单而失败，补充精确D723合同后通过。
两次失败均在GPU/远端启动前，原记录保留，不放宽身份或测量规则。

结果与收尾尚待实际执行。prelaunch目录保存资格、精确管理脚本快照和SHA。
通过后先备份再启动；已有V2分歧不强推，专用backup分支明确披露。
全部3B后续比较、消融、S1–S13、CE与审稿交付仍保留，当前不生成图。
