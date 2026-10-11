# D438：Loquetier 原生计量与受保护容量诊断

日期：2026-10-12。状态：原生观测CPU检查通过，单卡容量诊断完成并通过。
接续D437；不替代完整计划，不宣称Loquetier已通过正式性能资格。

## 修改依据与范围

既有gate缺少共同原生首末token、实际计算绑定和生命周期信息，属于R2。
本次只作正确性/计量/安全适配；不改变batch、精度、原生计算、静态分片，
不搜索性能配置，不使用未来工作量。
参考[官方仓库](https://github.com/NJUDeepEngine/Loquetier)的虚拟模块/SMLM定位，
以及本地固定HEAD `aae33baeeb19777129c1ccbff99a898d4a0e2c63` 的原生实现。
联网查阅固定SHA的raw URL失败；具体源码证据来自已核验的本地checkout，
不声称网络重新核验了其全部字节。

在现有 `prepare_ieee_tc_serverless_stack.py` 生成显式Loquetier计量源码视图：
只在 `ModelGeneration` 中插入登记、实际forward前、已有 `.item()` 后和
原生完成标记后的观察调用。去掉这些新增import/call后，AST与原实现完全一致。
其他原生文件只读链接，不修改历史三个兼容补丁，不替换SMLM或原生批处理。

现有 `ieee_tc_serverless_measurement.py` 新增独立Loquetier观察函数：

- 启动时核对全部静态名称、每个target module的index布局及SMLM A/B指针表。
  此处显式一次CPU读回，属于服务准备；不在每token增加CUDA同步/张量复制。
- 入队前检查CPU prompt IDs、外部request ID、mark、目标长度、固定greedy合同；
  未知/错误分片ID在原生index0回退和KV分配前被拒绝。整批验证失败不部分登记。
- 实际forward前检查分组LoRA索引、segment长度和原生root context。
- 在每个原生整数 `.item()` 完成后记录时间、真实输出ID和数量，拒绝重复、
  倒退时间及提前结束。该时间是CPU整数可用时刻，不是GPU kernel内部时间。
- 结构性绑定、内容SHA、数值正确性分别记录；前者不能代替后两者。

没有修改S-LoRA、ServerlessLLM或vLLM的调度/加载策略。历史执行helper身份
仍由各自原commit和manifest定义；当前公共helper新增代码不追改旧运行身份。

## CPU验证

新源码/原生观察9项测试通过（2.277秒）；与5项静态容量测试合并再次通过
14项（2.055秒），无skip。原生测试在现有 `loquetier_20260520` 中运行，
直接执行源码视图中的原生方法，CPU小张量/mock KV，不是模型GPU算术或性能证据。

覆盖AST保真、源漂移、视图不覆盖、实际1/2/256输出、混合批次完成/排序、
未知与重复请求的原子拒绝、错segment/adapter、错指针/布局、提前终态、
重复token及倒退时间。共同ServerlessLLM/vLLM/S-LoRA测量和部署回归
163项通过（10.995秒）；负面断流测试打印连接关闭日志，测试结果为通过。

CPU检查域：CPU2,3,26,27，memory.high3/max4GiB，swap0，GPU隐藏。
aux0001/0002/0003/0004峰值分别253644800/252047360/374460416/23818240B，
high/max/OOM均0，按InvocationID及实际空进程树结束。
旧147项保护校验通过；不覆盖旧论文图表、结果和用户修改。

## 容量诊断：运行前登记

在现有 `Loquetier_project/scripts/run_loquetier_trace_gate.py` 新增显式
`--native-capacity-probe`。必须通过既有主仓库gated launcher和独立watchdog，
实际模型进程先核验72/80GiB服务资源域，再导入torch/读取权重。

只运行一次以下容量诊断，不把它称为100请求或性能回放：

- 选择D437静态字节最大分片（并列取index最小），即分片0/125-ID；
  不是按请求热点选子集。完整四分片500-ID定义保持不变。
- 复用原7B基座和本地只读工件；完整核验该分片物化文件SHA，无下载或复制池。
- 沿用旧gate原生加载序列、FP16计算、PEFT默认原始表示和原生补齐。
- 建立全部125-ID原生模块/指针布局；分配batch2/context1024的原生KV。
- 记录各阶段实际allocated/reserved/peak、阶段和错误；不改参数自动重试。
- 成功仅表示该单卡权重准备及KV分配通过；不证明forward workspace、
  四卡并发HOST峰值、数值正确性、真实远程或完整W0资格。
- 失败保留，按实际阶段区分兼容/计量错误与资源不可行，不缩小500-ID池。

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d438_20261012`。
`run_capacity.sh`复用现有gated-launch，单一GPU0，预计新增磁盘2GiB；
不连接或更改174远端。D78/D80只读发布缓存不变。
容量诊断不是正式GPU-s/SLO比较，不用其局部数据替代完整500-ID四卡资格。

源码视图manifest SHA：
`4dc33282f5777e3462b42c887c22de82ffa00b0ef8ea2aac3af52c9f0ff9dd7a`。
观测后model_generation SHA：
`05eefcce371ca5689d7de5f1e577ba9e5317cf2391a583286f7334d49e3de385`。
helper SHA：`a4c116150db7f66f7d7d475cabdc3951095977235c520fc95ad7a5d104e203de`。
source adapter SHA：`2e7e0d2cdb4aa96b239bf08c88ed1a1b8175330631a67c2087074e95dc146e10`。
gate/probe SHA：`dc417b9d3847aba55afe4d942b4271ab3f2ca0e42bffce1f06b92354f0905c72`。
D437静态清单原SHA/原执行脚本身份保留在原commit，不追改。

## 后续主线

容量诊断后仍须复用公共HTTP、真实remote、物理生命周期计量，接入静态分片
在线到达/取消/失败并测试。按100请求、500-ID覆盖、原1000、单轮完整4000推进。
本轮尚不能将Loquetier填入合格主表；HydraServe、其余7B比较、必要PrimeLoRA
优化、3B、W1/W2/G2、全部消融/机制/敏感性仍保留。不能以局部资格结束总目标。

## 单次容量诊断的实际结果

执行代码为 BASE `28ffe7ed969c19d2d8683d282b9bda90c1f70d73`；
MAIN运行前登记 `400c3c8b1f358659dc8e8bf464ace4f6e24fa5e8`。
2026-10-12约01:07–01:09（Asia/Shanghai）仅执行一次原登记条件，
`passed=true`，没有重试、缩池、精度调整或baseline性能调参。

125个静态适配器的全部物化文件SHA核验通过，原生加载125/125；
128个target modules的名称/index/指针布局通过。batch2/context1024的
原生KV分配成功。阶段边界的PyTorch实测字节如下：

| 已完成阶段 | allocated B | reserved B |
|---|---:|---:|
| 基座加载 | 13476847616 | 13478395904 |
| 原始适配器加载 | 18946220032 | 18964545536 |
| 原生混合表示准备 | 23140786176 | 23152558080 |
| 注册与指针检查 | 23140787200 | 23152558080 |
| 两请求完整上下文KV | 24214530048 | 24494735360 |

最终allocated约22.55154GiB，reserved约22.8125GiB。
与D437静态accounted值24214509568B相差20480B；这只能说明本次
权重/KV阶段的预测吻合，不能推断forward、临时激活和workspace的峰值。
113次watchdog资源采样中GPU0 NVML用量最大24831131648B；
这是采样最大值，不声称精确瞬时峰值。其他三卡没有本实验计算进程。

服务cgroup实测peak7744462848B，主机可用内存最低105536823296B，
磁盘最低162913902592B；high/max/OOM/swap和保护中止次数均0。
服务与watchdog退出码均0，服务资源域已移除，结束时四卡各15MiB背景显示占用。
服务InvocationID `c0bb3150f32e4699a29530914aeb2d0b`；
aux0006 `39681dc94bd747cc8f96f6f8b7ed6f60`，peak85028864B，
核验真实空进程树后关闭；tmux仅在确认该pane正常结束后关闭。
后置保护检查147项零变化；原生Loquetier三个历史补丁和S-LoRA十个补丁
diff SHA均保持原样。没有远端操作、下载、池复制、旧结果覆盖或绘图。

原始结果及资源日志的SHA见
`paper_results/ieee_tc/loquetier/d438_native_capacity_summary.json`。
`runtime_qualified=false`、`performance_run=false`保持不变。
该结果不提供数值正确性、实际forward内存、四worker同时运行、真实remote
或4000请求的证据。下一步进入必要的原生HTTP/静态分片/生命周期适配及
既定资格序列，不重复本容量检查，也不把它用作PrimeLoRA的比较优势。
