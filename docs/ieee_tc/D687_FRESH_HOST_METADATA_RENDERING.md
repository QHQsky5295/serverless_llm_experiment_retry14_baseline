# D687：新鲜HOST清点的描述物化优化

2026-10-18。父版本5bb08ffb1af21acd50e2fea728335fcae6587a99；不是D685
批量pin查询的重试。D685/D686该接口方向已结束，不改变原生is_pinned路径
或GC策略。完整3B仍D682：3694/4000联合达标、U15896.253912 GPU-s；
独立数值正确性未知，G1/G2未闭口。当前候选尚无完整性能结论。

## 证据、假设与唯一改动

D684同版本诊断已观测8686次HOST清点、均值74.234ms；不能从请求时延
直接减去此数。D144/D202已消除同次重复清点及无消费者view表，D352/
D663/D675/D680已处理特定事务复用，D671已处理独立图复制，不再重做。
D329仍要求发现当前tensor替换、storage resize、共享、pinning和隐藏字段。

当前源码仍在无view表的运行时路径逐view生成完整层名称、dtype字符串，
重复读取device，并为维度检查物化shape。新假设：将描述格式化留在输出
边界，减少同一次新鲜清点的Python临时对象，不影响任何决策输入。

- 每个view当前device读一次，保留CPU/dense/2D/nonempty判断；ndim仍为
  实际张量当前维数，不查历史。pointer/capacity/is_pinned逐view照常查询。
- 每个view仍读取当前dtype，以原生dtype值形成每adapter集合，最后转换
  成排序字符串；不跨调用保存，不延长tensor或storage寿命。
- module/side/index分别传递；仅view描述或错误路径生成完整名称，报错
  文字和定位不变。完整资格view表不删除，运行时图schema不变。
- 不改控制/路由/规划/admission/生成/资源/SLO或九个IEEE公式，不合入
  批量CUDA查询、GC控制、跨调用缓存或其他候选。

官方依据：
[vLLM CPU优化说明](https://vllm.ai/blog/2024-09-05-perf-update)指出Python
对象处理可能阻塞推理推进；这里只采用问题定位方法，不引用其加速倍数。
[vLLM0.30 manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
提供当前注册映射，
[LoRA权重表示](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_weights.py)
保留dense/packed实际张量，不提供本项目跨调用物理状态不变保证。
[PyTorch2.13属性实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/csrc/autograd/python_variable.cpp)
中ndim直接读取Tensor维数。在线源码与本地实际版本分别记录，不混同。

## 验证与停止条件

1. CPU逐字段对照精确父函数，含完整/无view/请求范围/staged、共享、混合
   dtype、空集和错误定位；复用D329反例及现有HOST/owner预算检查。
2. 检查每view实际device与pin调用，不能仅用schema/测试通过宣布性能收益。
3. 复用现有native tiny fixture进行一次父/候选完整清点对照，不新建模型、
   LoRA池或trace，不运行旧pin候选，不筛长点、不反复运行寻优。
4. 若有正确性差异则修正原因或撤销；若原生开销不支持则停止，不把CPU
   测试环境计时外推CUDA owner收益。局部结果支持时才注册普通完整回放，
   依既定联合SLO与物理GPU-s评估，不凭局部百分比提前接受。

使用system-profile保持独立资格与系统性能分离，run-experiment复用资源门
和关闭入口。当前变更：gpu_monitor.py内清点表示；测试新增父函数对照；
旧FakeTensor增加真实API已有的ndim属性，不在生产添加缺字段兜底。
无生产插桩。未启动GPU/远端时不宣称原生资格完成。全矩阵、图表延期和
旧147项/用户修改保护保持，推理机150/100GiB及防OOM规则不变。

## 已完成资格，不等同系统性能验收

297项CPU回归执行通过（复用了既有测试，不将重叠检查称作独立实验）。
父/候选完整、精简、requested和staged图相等；共享/混合dtype/空集及错误
定位一致。运行时每view读一次device，不为维度校验取shape描述，仍逐view
读pinning；dtype文字由每view改为每adapter已观察dtype集合的输出。

唯一原生tiny对照复用既有8×32层、1536views，4类完整图一致；真实CUDA
注册/解除等9项原有pin路径检查仍通过。双方完整清点仍用Tensor.is_pinned，
不是直接binding或批量pin候选。原始测量字段native/direct分别映射精确父
函数/当前描述物化，结果内显式记录该映射，不与D462或D685数据混用。

| 同一原生微测 | 精确父函数 | 当前候选 |
|---|---:|---:|
| 调用数 | 20 | 20 |
| 平均完整清点 ms | 8.261667 | 7.179395 |
| 配对方向 | 参照 | 20/20较快 |

平均比0.869000742，即本次组件时间低13.0999%。不输出运行间CI，不把20次
调用当20次系统实验，不从D682时延减去此比例。没有删除任何测量。局部证据
支持进入**一次同配置完整W0验证**，不是已经接受性能收益；不追加微测/旧
prefix或GC归因，完整4000与458224tokens仍按原生成和共同SLO核验。

5906资格与67780原生均exit0，qualification/start均递归空关闭；峰值分别
600137728/84582400B，high/max/OOM/swap0。实际native服务已退出/GPU释放。
未访问远端、原共享k3s，不创建权重/trace。冻结前按github-sync检查后备份；
完整回放使用独占新目录、D682同配置和已发布真实远端，仍未知数值证明保持。
