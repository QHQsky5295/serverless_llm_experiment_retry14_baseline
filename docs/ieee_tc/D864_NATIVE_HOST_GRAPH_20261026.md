# D864：同次新鲜 HOST 图的原生遍历候选

2026-10-26。生产仍为 D857；D858 完整 3B W2 联合时延达成 3746/4000，
不是目标完成。D863 的 1000 请求诊断已封存，不重跑、不替代完整性能点。
本次只进行一个有界组件候选，暂无生产改动或性能收益结论。

## 新证据、历史与假设

D863 的31个 TPOT 违约窗口中，demand/source_snapshot/prepare_host/
pending_gpu_reference 占控制交叠89.61%，routing snapshot仅2.52%。交叠不能
直接当作可扣除时延；四类入口共同调用当前 HOST 物理图观测。

D747/D748已把每view属性融合为一个原生descriptor，但整个图的Python循环、
per-view调用/返回tuple、alias集合构造仍在串行控制线程上。新候选把**单次
完整图遍历**纳入原生调用，保留同一时刻的每view检查及相同图，不改变算法。

- D329的对象替换、resize、alias、pinning、隐藏字段反例仍必须发现；不缓存。
- D685/D686的延后pin列表和直接CUDA查询不恢复；每view即时调用原生
  Tensor.is_pinned，不保留Tensor批列表、不改GC、不释放GIL。
- D687的Python字符串延迟物化已失败；本次不恢复该补丁，dtype/名称仍按
  当前每view规则生成。区别是整图执行和跨语言边界，不是改格式化策略。
- D671/D675/D680/D747/D751/D857已接受的事务和表示改进保持；不重新实现。
- whole-owner异步、ctypes-stat和D860文件projection退化方向不恢复。

官方依据（本轮联网核查，不能借用其加速数值）：

- [vLLM CPU路径分析](https://vllm.ai/blog/2024-09-05-perf-update)：同步CPU
  对象处理可能阻塞生成；具体本机瓶颈依据D863而非文章百分比。
- [vLLM 0.30 LoRAModel](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)：
  clone共享底层权重，不能逐adapter简单相加或按对象ID缓存容量。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)：
  保留原生pinning查询，不以字符串/旧标记代替当前物理状态。

## 修改和预先限定的验证

只扩展已有native_host_descriptor.cpp和诊断witness；生产gpu_monitor不改。
沿用既有Torch2.13/CPython3.12构建方式，新增独占库，不覆盖D747生产库。
模型选择、requested/staged验证和输出scope由原函数保留；原生图必须同字段、
同排序、同alias分配编号，完整/无view/请求子图/staged均不能偷换含义。

先做CPU构建、逐字段及当前对象变化/错误路径核查；不重复无关广泛测试。
保持exact Tensor限定，不宣称支持Tensor subclass或其他ABI。若CPU通过，
最多一次复用既有backend-pin-check的实际CUDA状态与tiny图对照，计时涵盖
整次清点；不新增backbone/LoRA池/trace、不删长点、不重跑找好均值。
只有原生语义一致、总开销支持，才考虑生产接入和一次完整W2回放。
CPU一致不代替真实pin注册资格，组件速度不代替G1/G2或数值正确性。

3/4GiB、swap0、辅助CPU2,3,26,27、单编译任务；不与GPU回放重叠。
147保护项、用户manifest、现成工件、完整未完成矩阵均保留。不操作共享k3s。
HydraServe两模型三场景的隔离适配结果继续复用，不因异步回复重新部署。

PLAN d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
V2 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。

## CPU组件结果：通过，尚无速度/系统性能结论

独立库252008B；最终11项定向检查全部通过（0.333秒仅为测试程序时长，
不是清点性能）。完整/无view/requested/staged图逐字段一致；共享、offset、
packed缺失子模块、dtype变化、storage resize、替换/移除、独立返回容器、
无权重引用保留及错误输入均已覆盖。CPU环境CUDA未初始化。

前两次各执行70项检查（含复用和重复发现的既有测试，并非70个新实验），
均仅循环元数据的异常类型不一致：预期RecursionError，实际空RuntimeError。
第二次仅改变递归guard未解决，说明最初归因不足。进一步核查实际安装的
torch/csrc/autograd/python_variable.h发现THPVariable_Check的isinstance失败
抛出PyTorch python_error；通用std::exception处理覆盖了原Python错误。
最终按Exceptions.h原生restore语义修复，不删除检查或改成接受任意错误。
两次失败的库、源码快照、日志及资源关闭回执保留；第三次只跑受影响11项。
这是确定错误原因后的CPU实现修正，不是第三次性能采样或挑选均值。

最终受限资源域峰值1201975296B，high/max/OOM/swap0；旧147保护项和用户
manifest未变。沿用C++17构建出现既有上游bit-field及新增pybind字段可见性
警告，构建成功；原文保留，不将构建成功等同ABI跨版本保证。

本次GIL覆盖整个图，而非每view返回Python；即使组件更快，也可能改变输入
线程等待，因此不能仅由CPU一致或局部速度直接接受。下一只复用既有原生
pin/完整图微测核查，再决定是否值得一次完整W2。生产仍D857、现有D747
库不覆盖；没有新GPU回放、远端访问、共享k3s操作或正式图表。
全部两模型同版本矩阵、baseline/dLoRA、A1–A5、S1–S13、数值证明、审稿
解释与最终V2交付仍开放。

## 原生微测入口已核查（尚未执行）

复用backend-pin-check，新增显式host-graph-library；必须同时指定已冻结的
D747 descriptor参考库，排除旧Python实现充当当前参照。9种实际pin状态
直接经过整图入口，4种完整图逐字段对照；原8×32层fixture、20组交替调用
和全部样本保存规则不变。两个库均用源SHA绑定，不覆盖生产库或重编译候选。
新增3项接口选择检查及原8项接口回归通过，复用已完成11项CPU图检查。
启动使用已验证的系统Python监督、原Torch2.13被测环境、既有资源域/watchdog。
本轮只改独立资格入口和对应测试，无生产控制/计量策略变更。
