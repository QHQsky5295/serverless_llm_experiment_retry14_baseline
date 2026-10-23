# D762：相同内容分词器的共享表示适配

承接 D761 的原生 CPU 诊断及 D760 完整失败，当前不是 baseline 性能搜索。
唯一假设：对当前 500 个已发布工件，逐字节相同的分词器无需按 LoRA ID
重复构造；共享表示可消除这一路径的构造/延迟回收内存，而不改变输入输出、
LoRA 权重绑定、原生 RR、store、扩缩容和真实远程交付。

## 边界与依据

- 保留 vLLM 0.10.2、D755 物理设备编译缓存所有权适配及既有服务配置。
- 仅显式启用 `verified_published_base_v1` 时，在后端开始接收请求前检查
  实际 TokenizerGroup 源目录与许可工件的 tokenizer/config 文件大小及 SHA。
  使用既有不可变远端内容索引；请求仍由原 resolver 下载并验证完整工件。
- 支持当前已经核验的 JSON 分词器布局；异构分词器或其他资产不静默降级。
- 原生 TokenizerGroup 的 `enable_lora` 仅控制分词器选择。关闭的是该对象
  的逐 LoRA tokenizer lookup，不是 backend/engine 的 LoRA 权重功能。
  验证输入、预处理与输出实际共享该对象，且尚无请求分词器缓存。
- 不引入周期 GC、进程重启、权重预取、传输减字节或内存/超时放宽。
- 非 opt-in 的生成源码保持原行为；改动及身份随 source-view、backend-ready
  回执保存，不修改安装环境或历史执行结果。

已核对历史提交与 D761：此前闭口的是编译缓存写入所有权，不是这次分词器
表示问题。新候选有 D761 的实际内存增长证据，不重复已否决的参数方向。
原始依据：[vLLM 0.10.2 TokenizerGroup](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/transformers_utils/tokenizer_group.py)。
参考 [0.11.0 Processor](https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/vllm/v1/engine/processor.py)
的共同 tokenizer 设计，但不升级后端，不把后续版本当成本机修复的实测证明。

## 预登记验证与停止条件

1. 在 3/4 GiB、swap=0、CPU 2,3,26,27 的辅助域进行受影响单元测试。
   包含 opt-in/source 差异、内容不一致、额外/缺失文件、LoRA 权重开关保留、
   实际输入输出对象一致性，以及既有远端 resolver 回归。
2. 原生 0.10.2 CPU-only：复用 D761 的 128 次实际访问顺序，检查同步/异步
   lookup 均返回同一对象、LRU 不增长；复用全部 500 ID 验证同一性质。
   使用原工件中一个内容相同的独立原生 tokenizer 作 encode/decode 比较。
   不加载模型、不启动 GPU/远端、不生成工件或新性能负载，不主动 GC。
   仍采用 2.5 GiB RSS / 180 秒停止线，记录实际结果而不是预写通过。
3. 通过后，只做一次必要原生 mixed-LoRA 路径检查，再登记完整 W0 新执行键，
   接同配置 W1/W2。原 100/500/1000 整套资格不重复。独立数值正确性证据
   不由 CPU 分词器等价或 HTTP 成功冒充。
4. CPU 检查仅支持内容/表示命题；实际内存、失败率和服务结果由完整回放判断。
   若验证不支持假设，不堆叠补丁或更改评价尺子；保留失败并回到实验主线。

## 已完成结果

33 项受影响测试全部通过，原生 CPU 检查在 10.91 秒内完成。628 次同步及
628 次异步 lookup 覆盖 D761 原 128 次访问序列和全部 500 ID；逐 adapter
分词器 LRU 为 0，始终返回同一个原生对象。lookup 前后 RSS 分别为
870248448 / 870502400 B，增长 253952 B（约 0.24 MiB）。独立分词器的
8 个编码/解码用例一致；这不是模型数值正确性的独立证明。

辅助域 peak=599142400 B，high/max/OOM/swap 均为 0。进程 RSS 包含共享页，
与 cgroup charged memory 不是相同口径，不混为一项。没有主动 GC、GPU
模型运行或真实远端访问，未改权重、负载与历史结果。测试中的临时小型
HTTP fixtures 只验证原 resolver 行为，不是新的远端性能实验。

下一步复用现有原生 100 请求入口，**只验证本次改变的 mixed-LoRA 路径**；
不再接一轮 500/1000 资格。入口现有合法 prefix 规模为 100/1000/4000，
不为缩短检查额外重建回放器或更改输入合同。通过后完整 W0 使用新版本键，
不得以短检查或 CPU 结果代替它；D760 的 1478 失败仍完整保留。

当前状态：CPU 等价/表示检查通过，实际模型路径和完整性能尚待验证。
原始目录：`results/ieee_tc/serverless_qualification/d762_20261023/`。
