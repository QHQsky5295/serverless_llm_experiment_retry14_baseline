# D432：S-LoRA 7B 真实原生路径资格（预登记）

2026-10-11。只进行原 trace 前 100 请求资格；不作为完整 W0 排名。
D425 历史复用、D428 HOST 表示等价、D430 服务集成和 D431 审计已完成。
本轮进入实际 GPU 推理，不重复已封存 PrimeLoRA、ServerlessLLM、vLLM。

## 固定合同

- 原 7B trace 前100请求：29实际 adapter、17,369 target tokens；完整静态500池。
- fixed_length_greedy_v1、相同 canonical prompt、原生 SSE 整数 token 和后端事件。
- 原生 Python3.9.25 / PyTorch2.0.1+cu118，S-LoRA c1ddf488781ea7f551cd0bb07bfd097124c93411
  及既有十文件兼容修改；D430 source view 直接复用，manifest
  51c6f1da21cc5070a7d71e567c10564d3066605546602705c0b1c5d8b9277004。
- 四个 TP1 副本、round-robin、顺序初始化、swap、原生 slora 调度，非 BMM。
- 历史 April24 主点和 July18 fixed-output launch YAML 都是 total token 14000、
  input cap984、context1024、batch token1024；running_max_req_size1000 是原生
  API 默认值。本轮显式保留。不是经过完整1/2/3/4卡选择的最优工作点。
- D428 packed_views_v1：每层一个 pinned packed storage、原模块为 views，
  同权重/字节/算法，无跨 ID 去重。四副本 adapter HOST 数据下界40.625GiB，
  不是总服务峰值；保留72/80GiB、2GiB swap和实际持续防护。
- 复用 D78/D80 已发布只读压缩交付缓存，不重新发布。全500静态工件在共同通知
  后串行物化，再顺序启动；业务通知后60秒开始，不等 ready、不利用未来请求。
  CPU-only物化不虚构GPU占用，分配后全部持卡时间计入。真实传输/竞争/加载保留。
- 资格保护1800秒、原open-loop时序、原D394共同SLO，旧未知数值资格不改写。
- 原生 Triton磁盘缓存保留并在运行前记录；无预启动engine/GPU/HOST adapter。
- 唯一 raw：/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d432_20261011。
  唯一私有物化：/tmp/tc-d432-7bs；预测新增峰值20GiB，启动仍需≥150GiB。
  复用D423启动/远端管理/收尾脚本，只替换路径与明确S-LoRA配置；无新框架。

## 验收与主线

实际 worker 必须在共同服务域内；native ready绑500路径/原生参数/源码/真实GPU子进程，
native request绑相同worker。100/100正确合同、完整原生时间线、所有声明副本实际服务、
全池500次唯一传输关联、物理分配/退出/释放闭合后，才能称受影响路径通过。
只收到HTTP200不是数值正确性证明；独立参考复用与不一致另报。

运行完成后：实际GPU释放→核对本轮远端InvocationID后停止→保存日志/receipts/
远端证据→D431既有slora_resident_rr_v1审计→备份→缺失500-ID功能覆盖→原1000
兼容→一次完整4000 W0。失败保留，先定位，不盲目重启/OOM重试或更改SLO。

本轮预登记尚不是GPU结果。其他7B基线、W1/W2/G2、3B、消融/敏感性全部保持OPEN。
本次缓存授权已在计划2026-10-10修订记录；没有明文凭据入文件或Git。

## 依据

历史 launch YAML：baseline results/paper_experiments/03_main_comparison/
20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/shared_inputs/
llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_slora_dp4_tp1_launch.yaml，
以及18_eurosys27_v2_c5_smoke/20260718T090511Z...对应固定输出launch。
[官方固定API](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/api_server.py)
本轮HTTPS内容SHA与本地原生API均为504a1128cd8c6868a7d4c4ed21127b93f3cc7db0fc009082677bf1c1de97c485。
不是重新优化baseline策略；实际性能尚待观测。
