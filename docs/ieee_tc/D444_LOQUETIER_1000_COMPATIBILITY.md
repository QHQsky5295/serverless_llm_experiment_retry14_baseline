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
本轮尚未启动；先双仓库备份，再冻结输入与fresh资源/远端检查。

功能、计量和原生绑定通过不等于完整独立数值正确性或性能达标。D394共同阈值
不调整，原始数值未知标记保留；前缀不进入W0排名/参考，不套用D385原100参考。
下一步单次4000 W0，随后剩余7B基线/比较与必要PrimeLoRA优化，再3B。
W1/W2/G2、所有消融/机制/敏感性和审稿设计文档仍OPEN，完整计划未完成。
