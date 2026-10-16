# D557：vLLM 7B 高频轮换完整对照，运行前登记

2026-10-16。此文登记一次尚未启动的W2运行；实际状态以启动回执及终态证据
为准。遵循冻结指标V1、单轮V2和完整PLAN，不重跑D536或已封存的D554/D555。

## 实验问题与固定条件

补齐7B W2发表系统对照及共同常驻参考候选。输入复用D553批准的同一4000
请求视图：原arrival/prompt/target不变，仅按既定rotation100映射adapter，
497个实际ID、458224目标输出token。没有生成新trace、工件池或权重。
map SHA为48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300，
view SHA为5734bd88587c75dacc0d5c7534ad5b4d5d2a67aa0980f9b203dd87135d3adce7。

vLLM沿用D536冻结配置与原生入口，未作性能搜索：0.30.0、四卡常驻DP4/TP1、
round-robin、FP16、max_num_seqs2、max_loras4、max_cpu_loras16、rank64、
batch-token1024、context1024、显存比例0.7、chunked prefill开启、prefix cache
关闭、非强制eager。同一原模型与共同编译工具链不变。

源代码逐字节比较确认native入口、测量桥和native配置与D536相同。共享输入
入口仅具有已通过D553验证的W2扩展；实际退出计量使用D541已验证的pidfd poll
修复，避免旧高文件描述符测量缺陷，不改变GPU持有定义或补造释放时间。
复用386项W2检查、53项计量检查及D535共同软件构建；本次仅核查新输入接线与
原生配置合同，不重复GPU资格或旧性能实验。

首次离线复用检查把D554已验证的测试文件改动误记为test_ieee_tc_launch.py，
因而拒绝继续。原失败驱动、回执及实际关闭保留；核对历史发现唯一变化为
test_ieee_tc_preflight.py，SHA与D554资格记录一致。只修正复用检查的历史文件名，
新检查通过；未修改服务代码、评价阈值或native策略。

## 交付、资源与执行

共同部署通知在业务开始前60秒；不按各系统ready时间平移输入，未ready时
排队仍计入TTFT。fixed_length_greedy_v1、原生token计数和实际GPU分配/释放
规则不变。使用本轮独立空下载目录/tmp/tc-d557-7bv与原有远端只读gzip交付；
请求路径内不打包，真实传输、必要物化与竞争计入时延。

保留磁盘编译缓存，不额外加载模型或预热GPU。启动前记录已知编译前缀身份，
不声称证明整个底层缓存初态相同；运行内实际JIT、启动、排队与清理全部保留。
预期磁盘增长按32GiB检查；150/100GiB门槛、服务72/80GiB与swap2GiB、辅助
3/4GiB与swap0、既定CPU划分和单重型任务均不变。

先完成精确路径备份及实际关闭，再重新检查资源、保护清单、代码和输入SHA，
启动已有远端服务与本轮监控，使用tmux执行一次完整回放。远端只管理本项目
服务，记录实际协商带宽；共享k3s不访问、不重配。运行中不并行测试、构建、
哈希扫描、压缩、清理、备份或远端维护。

终态后依次核实实际GPU释放及递归空资源域、关闭自有远端、保存双边传输记录，
按既有resident_rr_v1/full4000_w2严格分析，再与D554/D555逐请求核对输入。
失败和计量缺口原样保留，不凭HTTP成功或token数量推定独立数值LoRA正确性。
尚未完成结果审计前，不冻结W2 U_ref/Q_ref、不宣布满足G1/G2。

原始目录：results/ieee_tc/vllm_qualification/d557_20261016；小型证据目录：
paper_results/ieee_tc/main_comparison/d557_vllm_w2。后续继续S-LoRA、Loquetier、
HydraServe W2，然后依据完整比较优化PrimeLoRA。3B、消融、敏感性与其他完整
审稿回应仍待执行，当前不绘图，不将一个场景视作整份计划完成。
