# D718/D719：3B 身份传播修正的原生验证登记

阶段仍为 PrimeLoRA 3B 优化；不重启已完成的 HydraServe 基础适配，不动共享
Kubernetes，不改变指标。D717 已通过逻辑测试，但没有新性能结果。本登记在
原生运行前完成；D718 是原 trace 前100请求功能检查，随后至多一次 D719
完整4000请求 W0。每个新键只运行一次，失败/退化保留，不按胜负重跑。

## 假设与范围

唯一生产差异是 D717：已接受的原生身份变更同步发布到既有路由 frontier，
避免后续重复使用已被推翻的 absence。前因、全量258次重试、反例测试、
vLLM 官方源码和 Redis 一致性文档见 D717 文档。此处不叠加第二个候选。
仅10个旧请求直接涉及该分支，其中1个联合违约；不预期它单独填补56条缺口。

D718 未触发罕见分支仍可证明一般路径未破坏，但不能证明全量重试消失。
D719 检查全量触发次数、相同否定观察重复、请求阶段分解与副作用；观测变化
不作为独立重复或因果隔离消融。版本仍是已知 trace 上的开发身份。

## 冻结输入与运行合同

- D703 全部模型、profile、autoscaler、batch、缓存预算与控制规律原样复用。
  YAML 差异只有 output_dir、nvme_cache_dir、host_cache_dir 三个独立路径。
  不采用已拒绝 D713 的缩容下限，不重新生成工件、trace 或准备 profile。
- Llama-3.2-3B-Instruct、vLLM 0.30.0、CUDA13.0 原生工具链、TP=1、峰值4卡。
- 真实远端 `prepublished_gzip_v1`，复用 D80 已发布只读对象，无请求时打包；
  不做额外远端配置/限速。运行前检查实际链路，运行中不扫描/压缩/清理远端。
- 原500逻辑adapter、seed42 W0 arrival/prompt/target；fixed_length_greedy_v1，
  CPU-only frontend，60秒共同部署通知，实际占卡至释放全部计量。
- 服务 memory.high/max=72/80GiB，swap≤2GiB；辅助4GiB，单一重型任务。
  推理机磁盘≥150GiB且满足新增32GiB峰值余量；不足100GiB安全停止。
- G1/G2、D648共同阈值和完整请求分母不变；无CI，不使用旧5000ms单指标别名。

源身份：

| 项目 | SHA-256 |
|---|---|
| PLAN | d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83 |
| V1 | 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22 |
| V2 | 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d |
| D648 SLO | d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8 |
| 生产 runner | 6f10f395fd74f01401c52f1e5698847ba2dfe704da32f883cf6aa933f86de6bd |
| D718 config | 8e41b4a78ade34032336a75a18d5e7b9ad4b6ecb82ca4488efe0fdeb02001406 |
| D719 config | 1d1a91b1fa3f41d764eb92e42f3da6210d3dd45d74203c9064fa31134ee10401 |

入口继续使用 `scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh d718/d719`；
每轮在独立tmux中，由原有gated-launch执行实际限制、外置回放和watchdog。
两轮原始目录分别为 `results/ieee_tc/p2_backend_qualification/d718_20261022/`
与 `d719_20261022/`。远端管理仅替换D713帮助脚本的campaign路径/monitor
身份；停止须核对实际release、服务InvocationID和端口，不操作不属本轮的进程。

## 验收与后续

1. D718完整100终态、原生token/adapter/content/lease绑定、无fallback；物理
   GPU release、完整dispatch字段及1ms时间重算门槛，资源保护通过。
2. 离线复用严格 `reevaluate_common_slo`，新增接受登记100前缀，不放宽其他
   字段或SLO；100/100不是4000/4000，也不代表数值adapter正确性已经解决。
3. D718通过功能资格后才启动D719，不要求短前缀先“获胜”。全量必须保留
   共同joint、边际达成、TTFT/TPOT/E2E、物理GPU-s和全部违约/失败。
4. D703当前参考为3744/4000=93.60%，14864.807562 GPU-s；n_correct和
   独立数值正确性仍未知，不能改成合格资源排名。D719是否达到性能要求由
   新数据决定，未知不补零，不因成功HTTP推断数值正确性。
5. 验证结束回到其余准备等待/短输出TPOT主因，不在此罕见分支继续局部搜索。
   3B W1/W2、其他baseline/dLoRA、A1–A5、S1–S13、CE/审稿文档/图表均未删除。

启动前：120项定向测试通过（9.00s）；JUnit SHA
`f72d8e667df3d6a797bb2697441ef9e938e35d11cb583b453ebe4b2767b820f8`。
144原投稿文件SHA零变化，用户manifest保留原dirty字节。preflight通过，SHA
`ffb5420e42c0676bba3139d8230694fbb379db8058aa03b53620c60b4ad33260`。
这些是启动资格，不是性能完成。本次不画图、不改论文源文件。
