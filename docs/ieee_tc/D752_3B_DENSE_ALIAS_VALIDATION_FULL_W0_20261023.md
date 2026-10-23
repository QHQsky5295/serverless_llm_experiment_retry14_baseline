# D752：紧凑分配所有权校验的完整3B W0验证

2026-10-23。REGISTERED；启动与完成分别由实际回执证明，登记不等于已运行。

## 问题与唯一差异

D749完成4000请求，联合达标3787/4000（94.675%），
U=14847.16496653203 GPU-s；尚未达到95%，也没有资源效率达标结论。
D750/D751已经给出历史、官方vLLM源码依据、单一假设、120次组件计时和
51项CPU正确性检查。本次直接推进到完整回放，不重复前缀、组件或GPU资格。

相对D749，唯一生产差异为D751的NativeSourceSnapshot._footprints局部
owner表示：连续allocation用数组，单owner用整数、真正共享才建set。
每次仍验证当前完整图/请求图、反向别名、容量、exclusive和pinning；
无跨调用缓存，不改变九公式、源/plan hash、路由、驻留、admission或控制参数。
候选代码SHA e604a7bf061db2da407c44195897cb83eacd586f2d79ba065eb24496ed26e039。

## 不变合同与资格复用

- 配置直接来自D749；仅output_dir、nvme_cache_dir、host_cache_dir三路径隔离。
- 原3B W0源seed42、4000请求、500逻辑adapter、固定输出/greedy和原生token；
  同canonical prompt、生成合同、D725 service/preparation profiles。
- D748/D747原生HOST descriptor和CUDA资格直接复用；精确library SHA
  b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044。
- 同vLLM0.30.0/CUDA130工具链、1–4实例、CPU/内存、60秒通知与1800秒保护；
  真实远端D80只读已发布gzip，不请求时打包、不重新生成工件或trace。
- 只补两项新的runner/config合同检查；D751的51项与D748的9项检查不重跑。
- 保护145项旧投稿文件/用户manifest；不stage用户manifest，不覆盖历史结果。

raw：results/ieee_tc/p2_backend_qualification/d752_20261023/
3b_dense_alias_validation_full_w0（同一个路径连续写入）。
curated：paper_results/ieee_tc/p2_backend/d752_3b_dense_alias_validation_full_w0/。
使用已有run_d293入口、受限服务域、独立watchdog、tmux和实际释放回执。

## 评价和解释

完整读取冻结V1与单轮V2，沿用D648共同3B阈值。单键只跑一次，不重复取优。
报告全4000终态/原生合同、联合及边际SLO、TTFT/TPOT/E2E、物理GPU-s、
自然扩缩容、准备/获取次数、资源事件和完整来源。对比D749直接父点，同时
保留D742/D703等历史，不以仅胜过退化候选作达标。变化属于已知负载上的
单轮开发观测，不称独立留出或显著改善，不输出CI。

D711独立数值adapter身份限制仍为n_correct=null、ranking_eligible=false；
HTTP成功/原生token一致不能自动消除此限制。即便联合达成95%，仍须如实
区分时延可行性、数值正确性和G1/G2完整目标。W0不能代替W1/W2或预算验证。

若候选没有系统收益，保留原始结果，依据完整指标判断撤销这一个局部变更；
不改阈值、不筛请求、不再跑同键追求获胜。运行中仅轻量监测，不扫描/压缩/
构建/测试/push或修改远端。结束先真实释放，再分析、备份和必要存储维护。

启动前再次验证推理机150GiB/102GiB准入和32GiB预计增长；运行磁盘底线100GiB。
预启动明确路径提交并推送唯一backup分支，无force，不声称V2已同步。
共享k3s不接管；HydraServe隔离实现及已完成结果复用。3B W1/W2/基线/dLoRA、
7B最终同版本、A1–A5、S1–S13、CE/Pareto、两轮审稿文档与最终图表均保留。
