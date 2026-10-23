# D754：3B ServerlessLLM并发编译启动失败，保留而不进入性能比较

原3B源负载前100请求、29个adapter、17,369输出token；真实远端只读交付，
原生vLLM0.10.2/torch2.8.0、原生store loader，min1/max4/target8/keep10。
复用D74模型配置及D414/D555测量；仅按既有D409路径允许原生图执行。

| 观测 | 值 |
|---|---:|
| 计划/到达/提交/终态/响应 | 各100 |
| 请求失败 | 0 |
| 实际ready并服务的实例 | 2 |
| 另两个实例 | 编译缓存FileNotFoundError，未ready |
| 条件平均/P95 TTFT | 182,557.529 / 234,915.445 ms |
| 条件平均/P95 E2E | 186,503.848 / 240,144.761 ms |
| 物理生命周期GPU-s | 1,315.835328411311 |
| 独立数值正确数 | 未知，不填100 |

以上是失败资格运行的条件观测，不是完整W0或PrimeLoRA的优势证据。
100条响应的生成合同、原生计时及E2E/TPOT重算通过；严格分析器在四实例
检查处拒绝，原检查未修改。两个健康实例完成全部请求不证明扩容路径完整。

四个独立TP1引擎日志指向相同
`~/.cache/vllm/torch_compile_cache/770f1086c5/rank_0_0/backbone`。
EngineCore744451/744453在各自临时文件rename时失败；744449/744450成功。
安装版PyTorch2.8的`CompiledArtifact.save(format="unpacked")`先删除目标
目录，再向其中展开文件；vLLM0.10.2按配置/代码hash和逻辑rank组织目录，
本机四独立引擎的rank均为0。与共享写入竞争相符，不归因于GPU吞吐或网络。

依据：[PyTorch 2.8原始实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/standalone_compile.py)、
[vLLM 0.10.2缓存组织](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/compilation/backends.py)。
D409历史7B同样共享目录但未失败；其成功不证明并发写入安全，历史有效结果不删。

382次资源采样：服务峰值52,631,592,960B，主机可用最低59,849,023,488B，
磁盘最低161,161,117,696B；high/max/OOM/kill/swap均0。Ray另有磁盘使用率
超过95%的原生告警；不把watchdog无告警写成所有组件无告警，也不改保护阈值。
实际GPU释放、service/launcher消失、loader恢复、三个本轮远端服务停止。
两端证据已收集；分析失败回执完整保留。145个保护项及三个用户修改不变。

原始根`results/ieee_tc/serverless_qualification/d754_20261023`，精简
`paper_results/ieee_tc/baseline_audit/d754_3b_compile_failure/{failure,manifest}.json`。
下一D755仅验证并隔离编译缓存写入所有权；保留编译、原生策略、冻结输入和
所有实际启动计费，不增加baseline参数搜索。
