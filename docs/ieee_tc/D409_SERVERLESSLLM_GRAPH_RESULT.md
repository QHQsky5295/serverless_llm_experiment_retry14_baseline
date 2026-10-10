# D409：ServerlessLLM 7B 原生图执行资格结果

2026-10-11。原始 W0 前100请求的单轮受影响路径检查已完成；这不是完整
4,000请求主比较，也不是可行配置或最优配置。冻结 V1＋单轮 V2 和 D394
共同 SLO 阈值不变。使用现有分析器，未增加 GPU 重复、权重或负载。

## 结果与适用范围

| 指标 | 单轮观测 |
|---|---:|
| 计划／到达／提交／终态／原生绑定合同完整 | 均为100 |
| 失败／实际逻辑 adapter／原生输出 token | 0 / 29 / 17,369 |
| TTFT 平均／Type-1 P95 | 63,240.115 / 111,377.364 ms |
| TPOT 平均／Type-1 P95 | 31.700 / 34.571 ms |
| E2E 平均／Type-1 P95 | 68,734.464 / 119,311.280 ms |
| router queue／service TTFT 平均 | 62,570.839 / 554.701 ms |
| TTFT／TPOT／联合 SLO 达成 | 0 / 100 / 0，分母均为100 |
| 物理生命周期 GPU-s | 1,033.029004 |
| GPU-s／绑定合同完整请求 | 10.330290 |
| 实际 HTTP 次数／线上字节 | 29 / 50,650,865 B |
| 内容核验后的物化字节／请求缓存命中 | 724,665,100 B / 71 |
| 请求路径打包 | 0 |

四个引擎实际服务；原生计时、E2E和TPOT重算最大误差0ms。GPU占用互斥项：
业务前210.459271，到达窗口734.031593，drain68.220819，清理20.317322 GPU-s；
总和等于总资源。实际 backend-ready 在部署通知后200.191–213.958秒，
业务仍按共同协议在60秒开始，没有等待 ready 后再启动到达序列。

结论是：该图执行路径通过本轮输入/adapter/计量/资源资格检查，但100请求
短前缀联合 SLO 为0%，不能称达标。解码观察值支持继续检查较长窗口，
不能把局部解码速度代替包含启动损失的整体 G1/G2。

## 唯一性能配置变化及实际执行证据

相对 D408 的公开后端字段仅 `enforce_eager=true` 改为 false。
保持 FP16、vLLM0.10.2、原生 store loader、RR、min1/max4/target4/keep10、
max-seqs4、max-LoRAs4、CPU-LoRAs4、token budget1024、GPU fraction.72。
准备60秒、验证保护1800秒、原始 prompt/arrival/target、冷私有 LoRA cache、
真实远端和共同资源包络均保持。D403调度队列修复未再次改变。

[版本文档](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html#enforce-eager)
说明 false 允许 CUDA Graph/eager 混合执行。实际四个引擎日志均记录
PIECEWISE、capture sizes[8,4,2,1]及图捕获完成；每个记录额外0.09GiB，
捕获耗时2/9/9/10秒。原生编译汇总分别约38.06–71.77秒，均计入启动。
日志证明完成捕获及混合路径服务，不据此声称每个请求的每个算子均在图内。
不存在静默 eager 重跑。原生 env_override 已设置 compile_threads=1，
没有添加新的线程参数，也没有并行安装/依赖构建。

原生编译缓存目录为
`/home/qhq/.cache/vllm/torch_compile_cache/1c637ca91b/rank_0_0/backbone`。
本轮四进程均有动态形状编译日志；共享缓存行为属于原生实现，不能将四个
进程的编译秒数相加当作关键路径。后续保留/复用编译磁盘缓存须明确记录；
它与每轮冷 LoRA 下载缓存不同，不可将缓存变化隐去后声称严格单因素因果。

D408 的0–99事后片段平均TPOT为63.880ms，本轮31.700ms，观察下降约50.4%。
但 D408 是1,000请求运行，后续到达和排队反馈不同；这里只是瓶颈假设的
描述性支持，不是严格匹配100请求对照或统计推断，不比较两轮短/长资源比。

## 输出与独立参考

逐条核验 source/prompt/target、原生 LoRA ID、内容 SHA、远端 transfer、
receipt及原生输出数量；原始未知数值正确性标志仍保留，不以HTTP200替代。
离线复用 D385 的独立 HF/PEFT 参考：100/100输入和工件绑定一致，覆盖四种
实际权重类，97条完整序列相同。其余3条分别为req_00018/00022/00059；
已有独立首次分歧位置及top2 margin为.015625/.015625/0，实际token在HF top2。
这些是有界数值证据，未将全部语义正确性设为true，也不以匹配率择优挑运行。
大部分工件为零/重复的限制仍在；500逻辑ID的功能覆盖复用D406。

## 安全与证据

- 执行main：`5a3dac21ffe52e0651c988584feecda003f1a64d`；baseline：
  `d6dcf14449e595f461bb933007b62c87423377d2`。
- 原始根：`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d409_20261011`。
- 分析：`paper_results/ieee_tc/baseline_audit/d409_7b_graph/audit_v2.json`，SHA
  `9c35cbb2661567d718f9671e605cf69b38099fdbcefbebbf9c688f2200db1218`。
  先前不含独立参考的audit.json仍保留，不覆盖。
- 请求原始日志SHA：`2c0893aa2101039ec26589b40f64eaef727c2e2384491ffaa5258783cd2a8967`。
- 同目录evidence_manifest保存126个保留文件，全部SHA复核通过；清单SHA
  `52c597fe13b0b0bb2782b1a533eba8d12a10a1b2ad5f2a26518107fac2a30fcb`。
  baseline收尾说明备份为`336576bc8a3e7aca34febad52372911289a9ca87`，
  已推送origin/main并核验远端SHA；不混同为本轮执行代码版本。
- 服务内存峰值50,855,706,624B；high/max/OOM/swap均0；主机可用内存最低
  60,230,864,896B；磁盘最低161,426,763,776B。Ray有磁盘使用率超过95%的
  原生告警；所保留raylet终态记录currently-spilled bytes为0，不改阈值掩盖告警。
- GPU实际释放、服务域删除、空aux域停止；原生加载补丁按原字节恢复。
  首次restore命令误指不存在的输出receipt，读取阶段即报错，未修改环境；
  随后使用原install receipt完成正确恢复，restore_receipt.json保留。
- 三个远端所属服务按InvocationID停止，端口消失；monitor/transfer原始文件
  已获取并核验SHA。链路1000Mb/s/full；复用D78/D80发布缓存，未重新发布。
- 147项保护检查前后相同，唯一旧差异仍为用户replay修改，未动该文件。
- 分析器15项CPU测试通过（0.052秒，4GiB/零swap资源域）。分析技能按
  观测—解释—后续检验组织证据；最新用户指令下不新增图或置信区间。

## 下一步

受影响路径已通过，回到原始1,000请求的公开配置验证，再进入完整4,000请求。
不继续扩展并发、形状或调度参数，不因0%短前缀联合值改变SLO或删除启动请求。
下一验证须登记原生编译磁盘缓存来源并保留全部实际启动成本。
PrimeLoRA D371封存点保持不变。全套7B比较、Resident预算参考、W1/W2/G2、
3B及消融/敏感性仍未完成。
