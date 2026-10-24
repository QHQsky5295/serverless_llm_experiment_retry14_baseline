# D803：HydraServe 3B单阶段模型数值检查

2026-10-24。Loquetier 3B W0/W1/W2完成并封存后，继续既定HydraServe
3B必要适配。复用原生修改后端、已有3B权重/LoRA、D790两个真实输入，
不生成新负载、不升级环境、不启动其他项目的Kubernetes。采用run-experiment
的现有受控启动和vLLM serving的模型加载诊断，不套通用性能阈值或调参。

## 结果及解释范围

| 检查 | 实际结果 |
|---|---:|
| 原生prefill / paged decode检查 | 2 / 4，全部通过 |
| 最大相对L2误差 | 0.0019939281 |
| 最小cosine | 0.9999924898 |
| 最大绝对logits误差 | 0.0322265625 |
| greedy token与HF/PEFT参考一致 | 6/6 |
| 前1024位置RoPE cache最大绝对误差 | 0 |
| embedding / head真实GPU参数别名 | 保持 |
| 实际模型参数dtype | 全部FP16 |

判据在执行前已固定为relative L2≤0.01、cosine≥0.999、greedy token一致、
RoPE cache误差≤1e-3，没有在看到结果后放宽。两个逻辑adapter ID为
finance_lora_0025与finance_lora，属于同一独立权重内容；两个prompt不同。
这不是两种独立训练权重的可区分性证明，不是完整4000请求正确性证明。

原生路径使用实际Worker/ModelRunner的输入准备、LoRA激活、cache和采样，
只在compute_logits处读取观测；未手工构造另一套slot布局。参考在原生
模型释放后，使用已有HF4.45.2、PEFT0.17.1、Torch2.1.2+cu121环境顺序
计算，不向HydraServe专用环境安装PEFT。原生GPU allocation峰值为
7,099,779,584 B，参考加载前残留allocation42,074,112 B。

通过说明本次受测PP1模型的RoPE、共享head、精度和LoRA计算与既有参考
满足预定误差合同。`pipeline_consolidation_qualified=false`、
`provider_shared_memory_qualified=false`、`full_workload_numerical_correctness_proven=false`
保持原值；不能据此宣称完整HydraServe复现或性能领先。

## 必要入口修正与历史保留

第一次CPU准备尚未创建资格结果、metadata或GPU任务，就因两个仓库的
`scripts`包名称冲突退出。D574历史已经记录过同类问题；本次没有再用
路径顺序临时绕开，而是复用现有`prepare_ieee_tc_serverless_stack.load_guard`
按具体文件加载安全模块，在准备、原生probe和参考子进程中统一使用。
[Python导入规则](https://docs.python.org/3/reference/import.html#packages)区分
regular与namespace package，解释了为何仅插入另一个根目录不保证选中它。

原失败源码按原SHA保留为prepare_import_failure.py，错误与精确空域关闭
记录于prepare_import_failure_closure.json；实际returncode1、无GPU/模型
执行。增加已加载baseline scripts包时仍选中正确安全模块的CPU回归检查；
三个相关测试模块共32项通过，0.942秒。没有修改原生调度/attention内核，
没有为低性能添加兜底，也没有用启动失败替代模型数值结果。

## 资源和来源

本次真实service/watchdog均返回0，GPU context已释放、服务域已消失，
probe辅助域也已按身份关闭。41个资源样本中，主机最小available为
106,053,029,888 B、服务峰值8,351,453,184 B；high/max/OOM/OOM-kill、
swap与abort均0。内存和磁盘规则未降低，没有改远端服务或其他项目k3s。

- 原始目录：`results/ieee_tc/hydraserve_qualification/d803_20261024/`。
- 数值结果SHA：`a8445cfbbf280707a08a270cca98bd5f1d80607ea9a0269c0b7b3cd596a8b1a5`。
- 原生logits SHA：`4fd6208ae10e8da883f8ecec76027c3a324bcb16e6ec3dde9ace8a09f8b0b444`。
- 保留输入SHA：`c81d3376854fcc9d8f6b8ebb431908ae49df86c6a2d5d08cc3eb7f8de43c598d`。
- overlay manifest SHA：`c17b4455f928317ba27fc1ac79cbee2314bebd7392e81e5a115adc9b451680fa`。
- 执行probe源码SHA：`a4e6640e7b285f626acc924267a73069256876a6e8da0dfe3ec2887db3bfa4c9`。
- 执行前BASE提交05be2d8加明确记录的入口导入修正；不把后置提交追改为
  执行前干净HEAD。资格manifest含实际源码SHA，原CPU31项回执仍保持历史字节。

使用analyze-results区分观测与未覆盖主张，使用github-sync作显式路径备份。
没有性能回放、SLO/GPU-s排名或CI；按用户最新指令暂不绘图。

## 下一步与主线

复用D576/D579/D582已验证的隔离平台、原生provider/shared-memory、HTTP
合同和完整场景入口；3B仅修改实际模型注册、元数据与此次必要模型overlay。
先核验受影响的权重交付/共享及流水线路径，再完成HydraServe3B完整场景。
不重做已通过且未变的平台资格、不盲加100/1000回放、不做baseline参数搜索。
之后是全基线比较、PrimeLoRA有证据优化、同版Full/A1–A5/S1–S13、
成本/CE/Pareto、两轮审稿文档、图表及最终V2；本检查不关闭完整目标。
