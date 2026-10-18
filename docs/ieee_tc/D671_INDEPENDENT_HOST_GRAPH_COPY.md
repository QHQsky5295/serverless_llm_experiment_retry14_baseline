# D671：保留新鲜状态与独立返回的HOST图复制

2026-10-18。延续D668实测瓶颈和D670组件验证，只实现一个候选；尚无新GPU
性能结果。当前3B完整W0仍为D665联合88.75%、15896.447805 GPU-s，未达目标。

## 改动及合同

`gpu_monitor.py`新增私有`_ieee_clone_registered_host_inventory`，只用于
`source_snapshot`没有staging时，复制刚刚实时生成的registered HOST图。
两个表及其中每个dict/list均独立，标量保持原值；空图同样独立。不改routing
快照、staged清点、allocator查询、每view的pinning/容量/表示核查、前端union/
exclusive检查、跨调用新鲜性、九个公式、策略或模型配置。

该函数只接受无tensor-view、无staging、非request-scoped的运行时投影。
顶层字段固定；记录只含原生不可变str/int/bool及其列表。新嵌套结构、容器
子类或不支持的字段类型明确拒绝；不是通用序列化器，不以浅拷贝兜底。
相比D670原型，另外显式验证dict/list及字段名类型，避免无意接受自定义容器。

`test_ieee_tc_gpu_references`中的一个旧mock原来只给字节数，没有完整图。
该fixture改为合法小型图，并保持动态字节与独立性断言；未放宽生产合同来
迁就不完整mock。新增5项测试覆盖值一致、所有可变容器隔离、空图、范围/
结构拒绝及真实调用分支。现有测试继续覆盖tensor替换、storage resize、
alias、pinning变化、非法表示、staged别名、allocator和执行引用。

历史依据见D329、D202、D663、D668及D670，不重新启用跨请求epoch/TTL缓存。
再次核查[Python复制语义](https://docs.python.org/3.12/library/copy.html)、
[vLLM 0.30管理实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
和[PyTorch 2.13 pinning实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)。
它们支持保留对象独立性与实际状态检查，不提供本机整体性能保证。

## 验证结果及限制

- 既有Torch2.8环境：685次测试执行，682通过；3项原生vLLM方法绑定测试
  因缺少`vllm.lora.model_manager`而导入失败。原失败日志与回执完整保留。
- 实际vLLM0.30环境：定向45次测试全部通过，包括上述3项及完整关键HOST
  图/新鲜性路径。部分测试类因既有模块导入重复执行，以上是执行次数，
  不是独立样本数或性能重复。没有修改生产代码以规避这3项导入失败。
- 两环境测试均禁用GPU，真实tiny CPU tensor加受控CUDA/owner fixture；
  不冒称真实GPU/RPC资格或端到端加速。D670旧7B图局部耗时也不外推3B。
- 实际资源域3/4GiB、swap0、CPU2,3,26,27；峰值分别617136128/638713856B，
  high/max/OOM/kill/swap均0，两域已按身份递归空关闭。
- 147项旧投稿保护文件及3项用户修改SHA未变。没有触碰非项目k3s，HydraServe
  已完成的隔离适配和结果继续复用，不因迟到的权限回复重新开始基线工作。

生产文件SHA为`0bec753ca9c5c04805f89940d83f75894a6c7f5b1a06d405d59389ce1cbea607`。
原始记录：`results/ieee_tc/p2_backend_qualification/d671_20261018/`。
本轮连同D670未发布组件证据、D669晚关闭回执在正常检查点备份；原始大图不复制。

## 下一步与完整范围

候选通过代码层验证，尚未接受为性能改进。备份后复用D665完整3B W0配置和
4,000请求，只更换新run路径、源码身份；详细core观察关闭，不叠加其他优化。
重新检查内存/磁盘/GPU准入、真实远端和同一生成/评价合同，单键只运行一次。
完整联合SLO与物理GPU占用决定下一步，不用局部组件耗时替代达标。

继续保留3B W1/W2/基线/dLoRA、同版本7B、A1–A5、S1–S13、成本/CE、两轮
审稿文档和最终图表；未达到目标前不画图、不调阈值、不把未知数值正确性升级。

`system-profile`限定这一实测热点和证据边界；没有新增生产计时仪器。变更：
`faaslora/memory/gpu_monitor.py`复制函数及单一调用点；上述两个现有测试文件；
raw资格/关闭/备份复用脚本和本说明/执行状态。`github-sync`要求显式路径、
源码/证据SHA、保护和凭据核查；备份回执以实际推送结果为准。
