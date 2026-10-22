# D728：3B无损backbone流水加载候选

状态：实现/资格中，尚无性能收益结论；不改变九公式或评价协议。

## 假设与历史

D726完整W0为joint92.650%、U15884.281128 GPU-s，未达95%。D727确认首请求
11244.304ms进入滚动P95，14.735/16.818/18.883秒连续触发扩容，而非当时
queue/active饱和。初始ready晚于业务8.924秒；所有启动等待仍计入TTFT/U。
已有worker日志中model loading10.678秒，weights8.42秒、prefetch1.16秒；
这些是嵌套阶段，不相加、不声称可全部消除。后续116条本地tier TTFT违约
和TPOT违约仍存在，启动改善不自动解决全部SLO。

已复核D338/D339/D340/D467：删除早期导入没有减少相同工作总时长；有线程的
前端不能fork；CPU前端预备已经实现。也不重开D685–689 HOST清点微优化。

可证伪假设：原生逐tensor装载的准备/搬运可由fastsafetensors流水加载减少，
在权重、dtype、计算图、LoRA、控制器及准备通知不变时缩短实际初始化。
先复用原生四请求生命周期资格，观察完整startup而非仅库内搬运。资格通过
后才考虑新执行键4000 W0；若不支持假设，不盲目重试或叠加另一候选。

## 官方依据与适配

- [vLLM 0.30 weight_utils](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/model_executor/model_loader/weight_utils.py)
  使用fastsafetensors ParallelLoader。其TP1默认尝试GDS，NUMA参数未显式传递。
- [vLLM公开模型loader注册](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/model_executor/model_loader/__init__.py)
  允许独立命名loader；不替换原生默认loader或安装目录文件。
- [fastsafetensors官方实现](https://github.com/foundation-model-stack/fastsafetensors)
  本机0.4.0 ParallelLoader公开set_numa参数。默认True可调用numa_run_on_node，
  不适用于本实验由应用限制的CPU集合。保留set_numa=False。

通过DefaultModelLoader子类只换权重iterator，原文件/index选择、parameter
赋值、dtype转换、post-load检查保留。固定nogds=True采用公开pread/bounce
buffer路径，避免cuFile初始化额外设备context；不声称测试了GDS性能。
队列沿用库/vLLM默认及其内存规划，不扫描线程/队列参数，不生成转换权重。
不重建LoRA/trace、不移动准备到部署通知前、不在服务外预持GPU。

只允许显式unquantized FP16 TP1 native owner。默认路径不变；未知或冲突配置
拒绝，加载异常不重试另一条路径。实际执行配置保留policy身份；post-ready
service/preparation profile可沿用同计算配置D725，不重测无关423个样本。
CPU测试检查配置、profile边界、prefix/selection、异常与关闭；native资格
复用现有gated-launch/worker计量与原四请求。资格并非独立数值正确性证明，
也不把历史n_correct=null追改。

使用system-profile、analyze-results、serving-llms-vllm指导边界和证据检查；
技能通用示例阈值不替代冻结V1/单轮V2。此时不画图、不操作共享k3s或远端。
