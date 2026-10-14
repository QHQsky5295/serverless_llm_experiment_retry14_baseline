# D525：原生请求映射的无损整数表示边界

2026-10-14。延续 D520–D524 同一元数据构造假设，不新增调度策略。

## 失败证据与根因

D524 的真实 worker 已安装 `cpu_live_mapping_v1`，固定两组 staging 共
99,040 字节，实际 pinned 占用由63,300增至162,340字节；原生2 GiB与全局
16 GiB HOST预算未变。但第0组warmup的14条请求全部失败，不能导出初始化
时间，也不能作为Full性能点。表层的 reservation 未释放报错是worker退出
后的结果；首个原生错误在 `_integers(index_mapping)`。

官方 vLLM 0.30 的 `LoraState.make_lora_inputs` 从 `np.int32` 数组生成tuple，
随后 `LoRAMapping` 仅转为tuple、不转元素类型。原辅助函数要求
`type(value) is int`，因此把合法的NumPy整数拒绝了。D525用实际安装的
LoraState和LoRAMapping产生输入，再调用封存D524辅助函数，准确重现同一错误。
三个相关安装文件与官方0.30源码逐字节一致；身份保存在资格回执。

这不是OOM、旧参数失效、论文公式冲突或可用的性能退化测量。D524完整失败
与双侧日志保留，本机GPU/资源域和远端服务已实际释放。

## 修改与不变部分

- 接受不可变tuple中的精确CPU整数，用标准整数索引协议无损转为Python整数。
- 不接受布尔、浮点、字符串、数组、张量或仅提供任意转换方法的对象；先判断
  整数类型再转换，禁止通过张量索引隐式读GPU。没有catch后回退到旧路径。
- 仍拒绝未知正adapter ID、重复物理槽、行数越界及非法capture组。当前slot
  布局每次重新解析，不引入缓存、过期状态或推测命中。
- GPU目标地址、图绑定、固定staging、DMA事件退休、原生激活/失效与所有预算
  不变。九公式、路由、admission、生成合同和共同SLO不变。
- 旧Python-int fixtures覆盖不足，新增真实原生producer测试，而非修改fixtures
  使它们绕过原生类型；覆盖重排、混合base/LoRA、空输入、全部14槽与槽位变化。

## 已完成资格及边界

393项CPU回归全部通过，零失败/错误/跳过，包含既有96组原生CPU metadata
oracle。CUDA未初始化。实际辅助资源域峰值2,312,175,616字节，high/max/
OOM/OOMkill与swap均为0；递归进程空且捕获路径已消失。147个投稿保护文件
未变。没有生成新权重、工件池或执行负载。

这是输入兼容与元数据算术/所有权的资格，不是完整模型数值正确性或性能收益。
D522设备算术证据保留但不重复跑；D524真实安装证据不能冒充成功生成。

## 主线下一步

D526只刷新受影响的初始化时间：同一个现有D298 runner、新的独立RAW目录，
原样引用D524已冻结config/spec（135组、1669调用），不复制/重建工件和负载。
新的源码提交与实际运行分别记录；D524失败不覆盖、不混入、不改名为成功。
验证实际publisher调用、固定pinned占用、完整生成/原生来源/时间类别和释放，
才可导出新的时间profile。共同warm SLO不重标、旧时间不借用。

随后新执行键原100 Full验证，再一次完整4000 W0判断资源与尾延迟。所有
7B W1/W2/G2、3B、消融、敏感性与两轮审稿交付继续保留。共享k3s不改动。

来源：[vLLM原生producer](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/lora_utils.py)、
[原生LoRAMapping](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/layers/utils.py)、
[NumPy整数类型](https://numpy.org/doc/stable/reference/arrays.scalars.html)。

RAW：`results/ieee_tc/p2_backend_qualification/d525_20261014/`。
汇总：`paper_results/ieee_tc/p2_backend/d525_native_integer_contract/`。
