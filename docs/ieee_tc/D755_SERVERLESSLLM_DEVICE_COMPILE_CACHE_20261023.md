# D755：ServerlessLLM独立TP1实例编译缓存所有权适配

针对D754两个引擎启动失败，而非为了改善基线SLO扫描参数。保留IEEE公式、
ServerlessLLM的RR、scaler、原生加载、batch/token/LoRA容量、生成和评价协议。

## 因果证据与最小改动

复核D409共享缓存成功历史及当前版本源码。安装版PyTorch2.8真实
`CompiledArtifact.save`方法在受限CPU测试中执行；仅将缓存载荷展开替换为
一个小文件，用两个进程控制交错。共享目录时A写入中的文件被B删除，A报
FileNotFoundError；独立目录时A/B均完成。真实删除/保存方法没有被改写。
这证明源代码层面的竞态可发生，不是GPU性能或完整模型正确性实验。

首次测试完成两个断言后，在记录模块来源时因import解析成同名函数报错。
原脚本、错误说明和关闭回执保留；第二版仅更正importlib模块读取及独立输出名。
不是重跑性能择优。第二版峰值1,012,838,400B，high/max/OOM/swap0，已关闭。

适配使用vLLM公开`VLLM_CACHE_ROOT`，在原生engine参数构造前按主机＋实际
`CUDA_VISIBLE_DEVICES`为独占TP1引擎分配稳定目录。vLLM原有配置/代码hash、
rank和prefix仍在其下，不自造编译键，不删除共享缓存，不关闭编译、不改eager，
不以run ID/PID产生永久冷缓存。首次编译仍发生在计费生命周期，后续合法磁盘
复用保留。适用范围是原生scheduler保证的每物理GPU一个独立引擎；不是通用
GPU共享/多引擎同卡解决方案。

显式opt-in保留旧source adapter的默认输出；配置身份写入manifest，实际
root/CUDA设备写入ready receipt。17项受影响测试通过，包括环境/配置不变、
不同设备独立目录、同设备稳定目录、源代码唯一差异、源manifest及ready证据。
原114项native与89项3B输入资格仅复用未变化范围，不冒称此次重新运行全部。

依据：[PyTorch 2.8保存实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/standalone_compile.py)、
[vLLM缓存目录规则](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/compilation/backends.py)、
[公开环境变量](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/envs.py)。

## 接续实验与边界

D756使用D754完全相同100请求和系统策略，只启用上述所有权适配；新的
`physical_device_v1`缓存根首次为空，其初态明确记录，不能将两轮时延差值
单独归为这项适配的性能增益。验收四个实际ready/served引擎、无编译错误，
再沿原计划补3B500-ID/1000兼容和完整场景。不降低原四实例检查。

不重做已完成7B、vLLM三场景或HydraServe隔离平台；共享k3s完全不动。
仍保留3B比较/PrimeLoRA W1/W2/G2、dLoRA、7B最终同版本、A1–A5、S1–S13、
CE/Pareto和两轮审稿文档/图表。原始及失败数据不覆盖。
