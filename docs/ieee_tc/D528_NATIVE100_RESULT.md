# D528：Full 原100请求完成，安装已验证，更新计数尚缺

2026-10-14。使用 analyze-results 的原始数据、解释与证据边界顺序。
源码执行版本9727bcfb71231c9cff25ef8ab114164b7f7cfb7d；输入、生成、
原生内容/引用/计时及远端双侧记录通过已有 analyzer。D527启动失败保留。

| 指标 | D475旧100请求 | D528新100请求 | 低优相对改善 |
|---|---:|---:|---:|
| 原生合同完成数 | 100/100 | 100/100 | — |
| 冻结联合SLO达标数 | 100/100 | 100/100 | — |
| 平均TTFT（ms） | 683.4874 | 689.7671 | −0.919% |
| P95 TTFT（ms） | 1493.6525 | 1512.1370 | −1.238% |
| 平均TPOT（ms） | 28.7586 | 28.8094 | −0.177% |
| 实际生命周期GPU-s | 334.0460 | 334.5301 | −0.145% |
| 原生输出tokens | 17,369 | 17,369 | — |
| 远端获取数/线上字节 | 29/50,650,865 | 29/50,650,865 | — |

1. 观察：两个实际worker分别服务90/10请求；均安装cpu_live_mapping_v1，
   pinned占用63300→162340 B，增量99040 B；处于原2GiB/native、16GiB/global
   HOST预算内。请求后两者均retired，HOST原生预留归零。1次initial及1次
   natural_scaleout，dispatch层remote/gpu/host/nvme为29/55/10/6，无缺失。
2. 证据边界：Full当前只保留安装时metadata snapshot，其updates均为0。
   这不是“机制没有触发”的证据，而是运行后计数未持久化；不能填造更新次数，
   也不能将D526的5060次标定更新作为本次运行的次数。安装/计账合格与实际
   使用计数合格分开，后者当前为false/未知。
3. 同合同：100条与已完成S-LoRA原始请求合同逐项核对request/adapter/target、
   canonical prompt SHA、原生prompt token SHA及source item SHA均一致。
   对D475同后端，90条输出token序列SHA一致、10条不同；ID见cross_contract。
   原D475对D471也有8条变化，不据此推断本轮变化无害。现有hash不能恢复差异
   token及其成因；不称数值等价，不无证据归为批处理/舍入，n_correct保持null。
4. 解释：该短回放未显示性能提升，单轮微小差别不作统计结论。它是执行/安装
   资格，不是最终G1/G2点，不据其胜负选择重复或删除结果。新标定和动态批次
   均可能影响轨迹，不能把阶段差额直接解释为单机制因果贡献。
5. 下一步：在已有原生完成/释放回执中保留publisher的CPU计数，不增加RPC、
   GPU同步、性能profiling或改变策略；补CPU回归并审计同后端数值证据边界。
   然后一个新键完整4000 W0，不重复本次100、D526标定、设备微测或已完成基线。
   完整运行中的更新计数是新增证据，不能回填D528；输出变化继续如实审计。

安全：服务峰值12652523520 B，high/max/OOM/kill/swap全0；真实GPU释放与
service/aux路径消失确认。远端两服务和监控inactive、端口关闭，双侧下载SHA
相符。请求中打包0，链路1000Mb/s/full，147历史保护文件及用户manifest未变。

原始结果SHA44933a4d08334f3a99a6f4217fca2fe23ca31c28f44924c8d60b746ed475665e。
curated：paper_results/ieee_tc/p2_backend/d528_metadata_full_qualification/。
完整7B W1/W2/G2、3B、dLoRA B6、核心消融、S1–S13及审稿文档仍待完成。
共享k3s未动；HydraServe D510的隔离适配与pp1覆盖范围继续保留。
