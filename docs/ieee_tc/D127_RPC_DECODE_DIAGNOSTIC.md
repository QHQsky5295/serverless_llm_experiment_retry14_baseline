# D127：消息解码的有界同输入检查

2026-09-30。开发期 CPU 诊断；没有修改服务代码，没有启动 GPU 或远端服务。
主线仍是修复 Prime Full 的大量排队与超时，不能将此组件测量当作 Full 改善。

## 假设与依据

D126 将成功请求 gate→外层终态的 11.438 秒占用上包络，分成源准入、准备、
原生生成和收尾；它并未直接证明哪条 CPU 路径导致长等待。D123 旧 GIL 采样
中 `_send_rpc_on_channel` 占 37,606 个样本，需要先确认其实际代码位置。

本次复用 D123 31 MB profile，只细分这一个已有类别：37,014 个样本位于
`run_all_experiments.py:5266` 下的 `json.decoder.raw_decode`，另 554 个
停在该行自身。原总样本为 279,902；这不是 D125 墙钟时间或等待比例。
源码显示响应在父事件循环同步解码，随后才处理完成、source 和所有权确认。

假设：保留完全相同的 JSON 线上字节及后续校验，替换消息解码实现可以减少
同步 CPU 处理。只测试解码，不改变 canonical SHA 编码、消息大小限制、
physical/source/epoch 校验，也不修改任何 IEEE 公式或调度策略。

参考：[Python asyncio CPU 阻塞说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
与 [vLLM 0.30.0 的 msgspec 序列化实现](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/serial_utils.py)。
vLLM 使用 msgpack/专门 tensor 编码，并不等于本项目可以无条件替换整个协议；
这里仅借鉴高效消息处理原则，用已安装 msgspec 0.21.1 做只读比较。

## 同输入结果

复用 D88 已核验 7B native source 组件：2 个注册 adapter、512 个 HOST
allocation；包装成同样的 RPC response 后 479,167 字节。不是当前最大
cache，也不是新的 500-adapter 完整实验。三轮交替顺序，每实现每轮 10 次；
每次解码后都检查相同 canonical 值 SHA，验证不计入解码时长。

| 实现 | 解码均值（ms） | 输入/解码后值 |
|---|---:|---|
| 当前 stdlib JSON | 5.515074 | 相同 |
| 候选 msgspec JSON | 2.583254 | 相同 |

平均减少约 2.932 ms（53.16%），仅适用于此已保存组件。保存每次观测和
ready callback delay；不计算工作负载 seed CI，不声称 TTFT 同比例下降。
大整数（含 2^100）、double、负零和合法 Unicode 额外检查通过。

候选对 NaN/Infinity 和孤立 Unicode surrogate 拒绝，而当前解码器接受；
因此不能称它在所有输入上无条件兼容。无效 JSON 的双方异常类型也不同。
需要明确原生消息合同与调用者错误处理后才可能集成；不使用静默回退解码。

## 失败保留与边界

第一次脚本将每个源容器统一限制为 24 MB，在读取 3B 的 253,310,434 字节
历史容器之前被尺寸保护拒绝，exit 1；未产出性能样本汇总。保留脚本、日志
和 3.77 秒/345,216 KiB 回执。没有放宽内存保护或读取巨大容器。

第二次只测当前问题对应的 7B 已有 20,341,207 字节容器；3B 明确为未测，
不造数据、不抽样挑优值，也不生成新的大文件投影。第二次总耗时 4.32 秒，
峰值 RSS 345,216 KiB，exit 0。两次均处于 3/4 GiB、swap0、CPU2,3,26,27
的实际资源域，high/max/OOM 等事件为零，精确身份核验后均已关闭。

## 决定与下一主线动作

**确认它是可减少的组件开销，但不接受为独立解决 Full 问题的优化。**
小型输入上的毫秒改进不能证明数秒级准备/收尾等待会消失；旧采样比例也不能
直接推算当前收益。当前不改线上 codec，不据此启动另一轮相同条件的 Full。
本项到此归档，后续只在更完整的控制路径优化确有需要时复用这些证据。

接下来检查同步准备规划与请求推进共用事件循环的依赖关系，界定哪些计算
只消费已冻结快照、哪些操作必须留在实时所有者中。只有能保持原计划选择、
当前物理重检查和取消/释放语义的方案才进入最小因果验证；不以增加请求
容量、线程总数、超时时间或减少检查掩盖排队。不要重复 D127 微测。

原始证据：`results/ieee_tc/p2_backend_qualification/d127_20260930/`。
完整数据将放在 `paper_results/ieee_tc/p2_backend/20260930_d127_rpc_decode/`。
Baselines/warm/Resident/M1/M2/A1–A5/S1–S13 均继续待办，整体目标未完成。
