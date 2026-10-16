# D560：最大静态分片的表示所有权与原生输出验证

2026-10-16。当前为已测试的诊断入口，GPU 结果尚未产生。
沿用 D438 的受保护容量入口，不改在线 serving、原生内核或工作点。

## 问题与唯一假设

D445/D544 的长等待主要发生在原生登记前；D438 显示原始 FP32 PEFT
表示与准备后 FP16 表示共占显存，batch2 已接近卡容量。D559 CPU 检查支持
进一步检验所有权，但不证明实际 GPU 内存释放或原生输出等价。
本次只检验：准备后表示独立存活时，纯推理能否回收原始对象，而不改变
共享基座、SMLM 指针和原生输出。尚不扩大 batch，不改分片和启动顺序。

## 方法

- 复用 D437 完整500静态映射，选择原定最大125-ID分片；FP16、batch2、
  context1024、apply_scaling A、官方虚拟模块/MixedLoraModel/SMLM不变。
- 显式 opt-in `--inference-owner-probe`；旧容量模式保持原行为。
- 输入复用 D445 worker0 第一条原生完成记录 req_00002 的前32个输入token。
  每个静态ID生成2个token，前后相同batch分组；这不是新性能trace或全负载资格。
- 回收前运行全部125个ID；记录原始对象/Parameter弱引用及独立storage字节。
  对共享基座及全部准备后权重/指针作有界CPU读回SHA和布局检查。
- 清空原始列表，一次部署期gc；不用cpu()/to()移动共享模型，不改dtype，
  不用empty_cache()伪装活跃tensor释放。实测allocated变化，reserved另报。
- 再检查弱引用、内容和指针，重复原生输出并逐ID完全比较；随后保留原有
  两份完整context KV分配检查。失败保留，不能直接提高batch重试。
- 当前源视图经原有生成器刷新，官方原始/测量源码SHA与旧视图一致；新身份
  只反映已有共享辅助代码更新。实际内核路径/SHA与单卡可见性再次核查。

## 已完成的前置验证

11项CPU检查通过，2.500秒，无skip。覆盖opt-in及来源SHA拒绝、全部ID/奇数
尾batch、错误绑定拒绝、真实虚拟/PEFT的复制和别名所有权、内容/布局敏感性。
峰值278552576B，high/max/OOM/kill/swap为0；真实资源域已关闭。
准备脚本前两次导入路径失败发生在GPU/视图创建之前；源码与失败保存，修正
诊断导入路径后通过。在线服务和已测试入口未因此改动。三个准备资源域均
实际递归空、关闭、路径消失；成功准备峰值36892672B，全部事件和swap为0。

## 来源与边界

历史：D437/D438/D439/D441/D445/D544/D559。固定官方commit
`aae33baeeb19777129c1ccbff99a898d4a0e2c63`。
[Loquetier官方实现](https://github.com/NJUDeepEngine/Loquetier)说明虚拟化与SMLM；
本机固定源码用于具体所有权关系核查。
[PyTorch 2.1内存说明](https://docs.pytorch.org/docs/2.1/notes/cuda.html#memory-management)
区分活跃tensor分配与allocator保留；因此不把nvidia-smi或缓存清理单独作为释放证明。

即使通过，本次仍只支持一个最大分片、短输入/两token的前后等价及实际容量
观测；不证明全context forward工作区、四卡并行、独立数值正确性、完整500
输出覆盖或性能改善。实际安全并发必须另据KV/工作区证据确定，之后新版本
按同一模型配置完整评估W0/W1/W2。原始弱性能与失败记录不撤销。
