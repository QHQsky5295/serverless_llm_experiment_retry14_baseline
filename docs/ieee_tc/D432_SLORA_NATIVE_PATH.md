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

21:28:49已启动实际资格，tmux tc-d432-7b；尚无终态GPU结果。
执行 main208db012ab8376fe696e7c43a368a233696169dd、baseline481a34ae0bea606372139e156e58daf0bee28d5d，
两仓库均已推送/远端SHA验证。24项CPU服务测试通过1.425s；原输入/源码/147保护项通过。
真实链路复查为1000Mb/s/full；只读缓存复用，无再次发布。
其他7B基线、W1/W2/G2、3B、消融/敏感性全部保持OPEN。
本次缓存授权已在计划2026-10-10修订记录；没有明文凭据入文件或Git。

## 依据

历史 launch YAML：baseline results/paper_experiments/03_main_comparison/
20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/raw/shared_inputs/
llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_slora_dp4_tp1_launch.yaml，
以及18_eurosys27_v2_c5_smoke/20260718T090511Z...对应固定输出launch。
[官方固定API](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/api_server.py)
本轮HTTPS内容SHA与本地原生API均为504a1128cd8c6868a7d4c4ed21127b93f3cc7db0fc009082677bf1c1de97c485。
不是重新优化baseline策略；实际性能尚待观测。

## D432 终态与离线核对（2026-10-11）

已结束，不重启。本轮100 offered/submitted/terminal/native/source/adapter合同全部通过，
零请求失败、17,369原生输出tokens、29实际请求ID集合中的adapter；四个声明副本均服务。
所有E2E/TPOT/native重算最大误差0ms。完整静态池500个adapter在四副本ready registry中验证；
这不等于500个ID均已发请求，后续500-ID功能覆盖仍需完成。

500次真实已发布gzip传输：420,740,410线上字节、12,985,984,450验证物化字节；
请求时打包0。共同通知后全池物化区间91.425097s；此值包含必要获取/本地物化，
不是纯网络时间。请求指向已准备工件，request transfer bytes0；500个部署传输只计一次。
四副本ready分别位于通知后206.032215/300.697412/401.555154/502.670611s，业务未平移。

| 短验证观测（非完整W0排名） | 数值 |
|---|---:|
| 平均 / P95 TTFT | 192,773.177 / 386,838.259 ms |
| 平均 / P95 TPOT | 44.741 / 54.815 ms |
| 平均 E2E | 200,318.923 ms |
| TTFT / TPOT / 联合SLO | 5/100、98/100、5/100 |
| 物理GPU生命周期 | 1,130.906283 GPU-s |
| GPU-s / binding-valid请求 | 11.309063 |
| 服务内存峰值 | 62,737,891,328 B |
| high / max / OOM / swap | 全0 |

该短验证不满足95%联合SLO。主要观察是路由/启动等待均值186.056s，
另有service TTFT6.697s，不能全部归为网络、纯计算或因果机制缺陷。
不得与PrimeLoRA4000请求结果相比较。资格路径通过不意味着性能工作点合格或最优。

复用D385原100条HF/PEFT独立参考：请求/input/adapter来源全部对齐，95条序列相同，
5条差异全部保留，其中3条可复用既有first-divergence margin、另2条未知。
不把它们一概归为舍入；跨后端精确一致不是V1门槛，也不宣称完整数值证明。
raw n_correct=null和独立数值资格标记未改写。

### 离线分析修正与验证

第一次完整审计在共同summary入口停止：它直接读vLLM native_lora_name，S-LoRA
实际记录native_adapter_path/index。运行时原生验证已显式支持S-LoRA，执行数据完整。
按冻结request_contract.native_backend添加明确分支：S-LoRA必须匹配绝对路径basename、
非负整数索引、S-LoRA timing/binding contract；不填造vLLM字段，不用替代名称兜底。
原vLLM分支不变，未知backend拒绝。错误路径、负/布尔/缺失索引、错误合同及伪造
vLLM名称均拒绝。37项回归通过2.005s；最初失败audit.log保留，成功audit2.log另存。
这是R1统计入口修正，不影响D432执行版本、不需要GPU重跑。没有性能算法修改。

D431既有严格slora_resident_rr_v1最终完整通过，包括独立publisher、原生进程/配置/
源文件、四物理分配、native退出/NVML释放、500传输/receipt/content/server关联。
新audit SHA ba4a963184497d6d2f1ad8f62640b54d552d113fdfbb3a59bb2a7a78921b894f。
evidence_manifest584文件/27,580,520B全部SHA复核，
SHA bf5ec1ba64e0f5dfc7f089602abb900312491ad6c1d63c3e44bb97a5678583b3。
位置：主仓库paper_results/ieee_tc/baseline_audit/d432_7b_slora_native_path/。

GPU全部实际释放、服务域已删除；本轮三远端unit按InvocationID停止、MainPID0/端口无监听。
147旧结果保护项零变化，三个用户dirty文件和S-LoRA十文件历史补丁未变。
下一步备份→必要的本轮可重建下载缓存回收/重新资源检查→缺失500-ID功能覆盖→
原1000兼容→一次完整W0；不能重复本100以挑更好SLO，也不跳3B或重跑已封存基线。

结果备份已完成：main fd32681c091d800f5eeedebb7021fbca34c7b3e2，baseline
15077ab4aeeac9532b1cb8d9f18567d76883163c，均推送并核对远端SHA。
随后仅回收本轮私有objects副本5000文件，释放13,004,439,552B；可用空间回到
163,332,403,200B。原池逐文件SHA/不同inode、记录副本、引用和挂载检查通过；
三个不可读同UID进程范围在回执披露。raw和远端缓存均保留，584项来源不变。
回收辅助域文件哈希出现memory.high回收，发生于GPU运行结束后；无max/OOM/swap。
无存活实验，下一项为既定500-ID功能覆盖，不重复100请求点。
