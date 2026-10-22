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

## 实际结果：完整观察，但不是完整性能点

启动版本35c9683af5cacbe46229a6655bc1d39a38bf8f06已先推送并核验
`faaslora_origin/backup/d723-components-20261022`。远端V2保持既有81990b7e…，
未强推，不声称V2同步。一次100请求回放完成；没有因结果重跑。

100/100通过原生生成/请求/adapter/lease/内容绑定和时间重算，独立数值
正确性仍未知。共同D648阈值下TTFT95、TPOT99、联合94；物理持卡
781.014570GPU-s。含启动与观察开销的前100视图不能外推4000或进入排名。
平均TTFT726.384494ms、P95 TTFT1825.217525ms、平均TPOT14.347116ms、
P95 TPOT18.125942ms；这些只界定本次诊断，不与无观察完整W0作收益比较。

4个原生进程、100个decode窗口全部观察完整，未截断；恢复11497个step。
下表是按实际分配形状分组的逐step均值，单位ms，**不是纯GPU阶段**：

|量|单token分配（11246）|含多token分配（100）|
|---|---:|---:|
|step wall|11.890723|62.501199|
|step线程CPU|4.380751|62.121280|
|execute_model wall|2.072450|60.418288|
|prepare_inputs wall|0.744666|0.911516|
|prepare_attn wall|0.174094|0.197156|
|_set_active_loras wall|0.328765|4.203630|
|sample_tokens wall|1.741254|1.418499|
|fullgraph主机调用|0.166428|0（0次调用）|
|piecewise_graph主机调用|0（0次调用）|0（0次调用）|

另外98个zero-token、53个no-schedule step完整保留。组件嵌套，不相加。
多token step中的execute_model线程CPU60.043774ms，scheduler0.297967ms；
因此主要主机成本不在scheduler、输入准备或LoRA激活，且这些已有组件并未
完整解释模型调用内部。没有GPU kernel观测，不能称GPU算力已饱和或GPU闲置。
进程allocator计数可能含后台free，不能将其累计时间从请求时延相减。

## 新定位：图重放覆盖边界，而非再次恢复元数据候选

全部4个worker的实际初始化日志均为enforce_eager=False，capture sizes
[1,2,4,8,16]、max capture16。100个多token step的scheduled tokens范围
90–764，每个都有一次execute_model，但fullgraph/piecewise入口调用均为0。
本机原生cudagraph dispatch在没有兼容候选时返回NONE，与这些观察一致。
这仅支持“所观察的图管理器重放没有覆盖这些step”，不把NONE误说成整个
模型完全未经torch.compile，也不把未覆盖段全部归为kernel发射。

下一有界问题因此是：**现有图捕获尺寸没有覆盖合法prefill形状，是否导致
额外主机模型调用成本；扩大按模型工作量推导的捕获范围，能否在计入启动、
显存与准备代价后改善G1/G2？** 先核对代码历史、官方图选择/捕获逻辑和
既有后端配置接口；最多一个有依据的模型级候选及原生正确性资格，不能
盲扫512/768/1024，也不直接改变调度token预算。配置改变须匹配合法profile，
SLO尺子不变；前100改善也必须经完整回放，不能推算获胜。当前未选择或
安装此候选，未改生产后端和九公式。D529拒绝结论不被覆盖。

短输出只有一条TPOT违约：decode287.432260ms，client交叠206.791303ms，
step75.324975ms。样本少，仍不支持“prefill能解释全部短输出违约”。
D719完整TTFT仅3791/4000通过，不能只修TPOT就宣称共同95%已满足。

## 收尾与来源

服务263次资源采样，内存峰值20011065344B，high/max/OOM/swap全0；
所有GPU已真实释放，服务路径消失，辅助域逐项清空。远端3B/7B与本轮
monitor已按InvocationID停止；真实29次传输、67350156线上字节，打包0，
本地/服务端传输SHA及字节逐项一致。启动链路1000Mbps/full，不改网络。
原生工作缓存已为空，磁盘仍约162GiB；没有删除唯一工件或原始证据。

分析峰值761556992B、high/max/OOM/swap0。旧投稿144文件与用户manifest
不变。无新观察代码、GPU同步、权重/trace/profile或正式性能点；不画图。

原始目录：`results/ieee_tc/p2_backend_qualification/d723_20261022/`。
精简证据：`paper_results/ieee_tc/p2_backend/d723_step_components/completion/`，
包含共同指标、完整观察摘要、组件表、实际图配置日志来源、管理/分析快照
及SHA。全部3B后续比较、消融、S1–S13、CE与审稿交付仍保留，目标active。
