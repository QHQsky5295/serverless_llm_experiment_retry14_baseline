# D825：受管远端物化只做一次最终文件内容校验

2026-10-25，CPU资格前登记；父版本9ff11900cd10b4538a8c140b206ea239d8b34a87。
不是已接受的端到端性能改进。继续3B W2主线，不改公式、控制、SLO、预算或工件。

## 证据、历史与可证伪假设

D817的Remote513请求中267条TTFT违约；D818将503条唯一传输分解到本地
extract/verify和publication包络，不能把这些包络全称为网络或SHA耗时。
当前HTTP路径解包时对每个未压缩文件hash一次，LocalSourceReferences发布
时又读取实际目标文件完整hash一次。D627实际32MiB文件测得后一次SHA
update约82–90ms，但这不是当前3B在线开销或预期服务收益的测量。

假设：保留发布前实际目标文件的完整SHA，只去除先前解包流上的重复SHA，
能减少远端物化的CPU工作，不改变确认时点、文件内容与安全/容量语义。
组件必须证明每个文件仍完整hash一次，且该hash在owner锁内、可见之前；
并检查同尺寸错误内容、取消、路径/大小/集合、已有副本和真实预算保护。
如果组件无可测工作减少或正确性不成立，停止，不启动GPU回放。
通过后仅具备一次新执行键完整W2评估的依据；必须实测联合SLO与物理U，
不能从阶段相减预测达标。旧D711数值限制及N_correct未知继续保留。

区别于已否决D628/D629：不移动最终hash或锁、不增加并发、不改变读者可
前进性；D631回退后的owner事务原封不动。也不重启D595/D598、D712/D713、
D822已否决方向，不更换控制阈值、共享压力通道或重跑同键寻优。

## 官方依据与适用边界

2026-10-25核查：[ServerlessLLM原论文](https://www.usenix.org/system/files/osdi24-fu.pdf)
强调多层加载路径效率，支持检查实际物化工作，不证明本改动或本机收益。
[官方store实现说明](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/sllm_store/README.md)
使用加载专用布局、内存池与并行I/O；本轮不移植其格式、不重建工件、不改baseline。
[HF v0.34.4源码](https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.34.4/src/huggingface_hub/file_download.py)
在私有incomplete文件下载后才发布；只借鉴发布边界，不声称其具有本项目的
物理预算/目标内容证明。[hashlib文档](https://docs.python.org/3/library/hashlib.html)
说明update对实际输入字节计算摘要，大块调用释放GIL；不能把GIL释放等同
owner锁释放或推断GPU推理收益。

## 实现范围

- 默认HTTP调用仍为stream完整校验；baseline路径不变。
- 只有具有workspace、reserve_files、publish_verified的受管调用显式选择
  owner_publication；无未核验fallback。它是验证责任合同，不是调参选项。
- 归档SHA、tar路径/类型/大小/完整集合、取消和预分配仍保留。文件签名只
  绑定关闭writer后的对象，不能代替内容摘要。
- 原owner仍在发布前读取目标、完整校验SHA，再rename并登记confirmed。
- HTTP证据在owner返回匹配内容身份的confirmed回执前保持content_verified=false；
  单独记录stream_content_verified，不冒充流上已验证。
- 不改HOST copy、本地cache、admission或runtime；不改变九公式和线上配置。

## 有界验证与资源

使用system-profile的包装式测量和既有D605/D627受限组件入口，复用实际
32MiB源文件；不复制整池、不生成权重/负载、不访问远端、不启动GPU。
默认/owner两路径使用同一tar成员流；若使用原文件提供成员流，则明确不
测gzip/网络。计时分开extract与publish，不将嵌套SHA再相加。全部值保留。
CPU2/3/26/27、3/4GiB、swap0；实际磁盘/内存检查、147保护集合、资源结束
核查继续执行。先受影响HTTP/owner/传输生命周期测试，不重复广泛后端资格。

改动清单：HTTP提取与证据合同、runner受管调用、两个既有测试文件；测量
使用外置组件脚本，无线上profiling。结果、资源和接受判断将在完成后追加。
最终同版两模型M1/M2、A1–A5、S1–S13、成本/CE、文档/图表/V2同步仍全部保留。

## CPU资格与组件结果

HTTP 37项通过；请求生命周期222项与传输/准备187项通过，共446项不同测试。
另单独运行的32项publication测试是上述222项子集，不重复计入。首次新测试
将每次捕获时间也当作不可变source状态比较，导致1项断言失败；改为比较
实际sources与epoch后通过，生产代码没有因此放宽。中间一次复用脚本的
phase替换未到达底层入口，在测试开始前退出；失败脚本和来源保存，改为
直接复用D595入口及D610逐模块隔离，不覆盖首次测试日志。

已有32MiB工件33589802B，固定ABBAAB次序；不是独立推理重复，不算CI。

| 调用 | 校验位置 | extract ms | publish ms | 合计 ms | SHA输入字节 |
|---:|---|---:|---:|---:|---:|
| 0 | stream＋owner | 111.463 | 103.127 | 214.589 | 67179604 |
| 1 | owner | 24.399 | 92.329 | 116.728 | 33589802 |
| 2 | owner | 25.447 | 93.292 | 118.739 | 33589802 |
| 3 | stream＋owner | 112.147 | 91.178 | 203.325 | 67179604 |
| 4 | stream＋owner | 110.635 | 91.222 | 201.857 | 67179604 |
| 5 | owner | 26.786 | 90.531 | 117.317 | 33589802 |

线程CPU时间与wall接近，逐调用值全部保留。唯一完整payload SHA仍在owner
锁内、目标可见前执行；内容身份相同，原文件和147保护集合未变。此组件
把现有文件作为tar成员数据流，没有gzip解压、HTTP、在线竞争或GPU推理。
所以约202–215到117–119ms只描述该组件，不能等比例预测当前3B请求TTFT。
无新图，不把此改动包装成论文新算法/贡献或已经解决Remote整体差距。

组件memory.peak347521024B；最终相关回归568651776B，high/max/OOM/swap
均0；初次失败也没有资源越界。所有CPU scope结束后保存实际关闭核查。

结论：局部工作减少和正确性条件成立，候选可以进入一次新键的完整3B W2。
尚未接受为性能优化；下一主线复用D817的输入、共同SLO、资源/控制/远端
合同与max3配置，只改变本候选源码。先看全部4000请求的联合SLO、U、尾
TTFT和失败，不追加短Full或按结果重复，不把旧W0/W1与新W2拼成同版结论。
默认pressure仍native_rpc_v1，不启用D822否决候选。D711独立数值限制仍在。
