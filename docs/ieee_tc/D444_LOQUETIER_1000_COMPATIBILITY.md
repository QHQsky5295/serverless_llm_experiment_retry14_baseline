# D444：Loquetier 原始1000请求兼容验证

日期：2026-10-12，预登记。D442原100和D443全500-ID覆盖已完成，不重复。
本轮使用同一冻结Loquetier配置，验证原始1000请求/60个adapter/122510原生
输出token；完整500工件池保持可用，不按未来热点缩池，不优化基线性能。

输入view SHA `0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981`；
native配置 SHA `d9f3b81d3d90340739ec66b45d36c1c71c706640dd2cd790a19d3ea4a80a617a`。
沿用D441原生SMLM、四125-ID静态分片、FP16/batch2/context1024和顺序启动；
notice后完整池准备、notice+60秒业务、既有真实远端只读发布协议不变。
反复确认的缓存授权仍只用于复用D78/D80，不生成第二套缓存或解压池。

完整预登记与执行配方见BASE `docs/ieee_tc/D444_LOQUETIER_1000_COMPATIBILITY.md`。
raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d444_20261012`；
private：`/tmp/tc-d444-7bl`。strict analyzer为`loquetier_static_shards_v1/prefix1000`。
53+18项CPU检查通过，D443的578来源文件SHA及147旧保护条目不变。
预登记时尚未启动；下述实际启动已完成双仓库备份、输入冻结与fresh资源/远端检查。

功能、计量和原生绑定通过不等于完整独立数值正确性或性能达标。D394共同阈值
不调整，原始数值未知标记保留；前缀不进入W0排名/参考，不套用D385原100参考。
下一步单次4000 W0，随后剩余7B基线/比较与必要PrimeLoRA优化，再3B。
W1/W2/G2、所有消融/机制/敏感性和审稿设计文档仍OPEN，完整计划未完成。

## 实际启动（运行中，不能视为已完成）

2026-10-12 03:24:00Asia/Shanghai单次启动，tmux `tc-d444-7b`。
MAIN执行版本`ebc58ffc380e0d1ed8194754464e8aa50d53d38a`，BASE执行版本
`12fefd5881f5216f868304d0e6ada98e90125bba`，均已推送且远端SHA一致。
服务`primelora-tc-svc-47f7021f111d449cbbe7b04e64be8933.scope`，InvocationID
`dfa0b37ecf54463aac8cb07c3eecffab`。外置保护已核验实际72/80GiB、swap2GiB。
请求HTTP合同SHA `04f34657972075220b4b14183efcb2b5fc198c9e893545d89658762592621a4e`。
真实远端启动前为1000Mb/s/full；两健康检查通过，未调整网络、打包或发布。
当前只监测此运行至结束，不调整配置，不启动第二重型任务；实际终态、资源
释放及严格证据分析之前不报告最终完成数、SLO或GPU-s结果。
