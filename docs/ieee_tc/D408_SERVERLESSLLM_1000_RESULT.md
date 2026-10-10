# D408：ServerlessLLM 7B 原始 1,000 请求验证

2026-10-11。状态：执行及离线核验完成，兼容性通过，联合 SLO 不达标。
这是已知 W0 的开发前缀，不是完整 4,000 请求主比较，也不是配置最优证明。
保持冻结 V1＋单轮 V2、D394 共同阈值；不新增重复运行或置信区间。

## 实际结果

| 指标 | 单轮观测 |
|---|---:|
| 计划／到达／提交／终态／原生合同完整 | 均为 1,000 |
| 失败 | 0 |
| 实际逻辑 adapter / 原生输出 token | 60 / 122,510 |
| TTFT 平均／Type-1 P95 | 4,683.001 / 36,802.780 ms |
| TPOT 平均／Type-1 P95 | 63.562 / 68.849 ms |
| E2E 平均／Type-1 P95 | 12,420.358 / 42,585.831 ms |
| TTFT / TPOT / 联合 SLO 达成 | 828 / 800 / 674，分母均为 1,000 |
| 生命周期物理 GPU-s | 4,584.427947 |
| GPU-s / 合同完整请求 | 4.584428 |
| 真实 HTTP 次数／线上字节 | 60 / 74,953,014 B |
| 核验后的本地物化字节／请求缓存命中 | 1,474,429,899 B / 940 |
| 请求路径打包次数 | 0 |

四个真实引擎均服务了请求；scheduler 收尾读回时仍运行、无异常、无待分配请求。
完整 E2E、TPOT、原生计时重新计算最大误差均为 0 ms。
GPU-s 四项为：业务前 145.309336、到达窗口 4,317.657120、drain 100.983792、
清理 20.477699；其和等于总量。原生 store 的四卡实际持有全部计费，
不能用逻辑副本减少替代物理释放。

1000 条记录逐一通过 source/prompt/target/native adapter ID、接收内容 SHA、
远端 transfer/归档 SHA、本地 receipt 和原生计量的关联检查。独立数值证据
复用 D407 的有界审计；不把本轮计数或 HTTP 200 当作独立数值证明。
原始 `n_correct=null`、独立数值标志和旧资格标志不改写；新的 `common_slo`
明确使用已验证的绑定合同。500 逻辑 ID 功能覆盖复用 D406，不重跑。
大部分工件权重为零／重复的限制仍保留。

## 为什么仍不能直接进入“达标基线”排名

联合达成率 67.4% 低于共同 95% 要求。以下是事后解释用的、覆盖全部请求且
互不重叠的源顺序分段，不是独立重复，也不从主分母删除启动请求：

| 源请求索引（含端点） | 平均 TTFT | 平均 router queue | 平均 service TTFT | 平均 TPOT | 联合 SLO |
|---|---:|---:|---:|---:|---:|
| 0–99 | 35,417.453 ms | 34,363.564 ms | 563.125 ms | 63.880 ms | 7 / 100 |
| 100–499 | 763.696 ms | 347.191 ms | 404.611 ms | 63.194 ms | 314 / 400 |
| 500–999 | 1,671.554 ms | 1,256.309 ms | 404.973 ms | 63.793 ms | 353 / 500 |

实际 backend-ready receipt 在共同部署通知后 167.977–168.029 秒才出现，
业务按协议在第 60 秒开始；第一个请求 TTFT 110.236 秒。启动积压确实存在。
但后两个区段仍有 TPOT 和联合 SLO 违约，因此“全是启动”也不成立。
全体平均 router queue 为 4,203.387 ms，service TTFT 为 420.643 ms。
分配间隔中位数 0.638 秒不能直接解释为服务容量；该统计也不是原来的
无条件逐请求一秒等待的因果效应估计。

不将这个前缀的 4.584428 GPU-s/request 与 PrimeLoRA D371 的完整 4,000 请求
直接排名；短前缀启动占比不同。保持 D371 不动，继续基线公开配置验证。

## 下一项有证据的配置问题（D409，尚未运行 GPU）

现有资格 helper 固定 `enforce_eager=true`；此次实际配置亦如此。
ServerlessLLM 固定官方源码允许配置该项，默认 false；vLLM 0.10.2 文档说明
true 会关闭 CUDA Graph，false 使用 graph/eager 混合执行。
因此，持续约 63 ms/token 的现象支持检查执行模式，但尚不能证明它就是原因。
参考：[ServerlessLLM 固定源码](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/backends/vllm_backend.py)、
[vLLM 0.10.2 参数](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html#enforce-eager)。

只增加显式 `--no-enforce-eager`；默认旧行为不变，不改 RR、扩缩容、原生
快速加载、LoRA 数量或公共阈值。先验证新执行路径的原生计量、adapter、
显存及启动代价，再用同一原始 1,000 请求验证。保持 target4/min1/max4/keep10、
序列上限4、LoRA上限4、1024 token budget、FP16、.72显存比例等不变。
新配置须有新执行键；不重复 D408 寻找更好的随机运行。图捕获失败不可
静默回到 eager 后仍标作 graph；新增编译/准备资源仍计入生命周期。
这是一次公开配置机会，不宣称“官方最优”，也不直接移植 PrimeLoRA 机制。

若改善假设被否定，保存结果、收窄解释并返回完整 W0；不在基线局部无限调参。
共同 60 秒准备、95% 联合 SLO 和既有内存／磁盘护栏不变。

## 安全、复用和来源

- 执行 main：`242b3c455e13ac10920f0af8a04695af05f52a7f`；baseline：
  `25848cfb57ef15351f800daf7062cffb2a0d3807`。
- 后续 D409 公开配置入口备份为 baseline
  `d6dcf14449e595f461bb933007b62c87423377d2`，已推送并核验远端分支SHA；
  它不是 D408 的执行版本。189文件来源清单SHA为
  `dbaa0ab5343027b29b0bcc964c1601a8d876f42c80fad63d1dd71ba220ed06b3`，
  全部重新核验通过。
- 原始结果：`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d408_20261011`。
- 内容核验及引用检查后，仅回收 `/tmp/tc-d408-7br/remote_artifacts/objects`
  的600个可再生下载文件；原始池逐文件SHA重核、inode不同，原始日志及
  resolver记录保留。回收前后回执在raw目录；观测释放1,476,124,672 B，
  磁盘可用恢复162,219,474,944 B。缓存字节再次审计需重新物化，不能因此
  重下载整个池或重复已完成实验。下一次启动仍须重新通过完整资源门槛。
- 审计：`paper_results/ieee_tc/baseline_audit/d408_7b_compatibility/audit_v2.json`，
  SHA `3a159e40315aa5291e70972b621dd77a1222f46c0b308785022c514cfcb28ca4`。
  先前 `audit.json` 原样保留；v2 仅增加 TPOT 汇总和源顺序诊断。
- replay SHA `1a9914e230481f13654955261ae1a77c44a9f476e1575792e7fc6a382e193449`；
  D394 freeze SHA `2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3`。
- 服务内存峰值 52,653,588,480 B；high/max/OOM/oom-kill 均0，服务swap峰值0；
  主机最小可用内存 59,081,170,944 B。运行中最小磁盘可用160,826,478,592 B。
- GPU 实际释放、服务资源域移除；overlay 恢复原像；远端三个本轮 unit
  按 InvocationID 核验后停止，MainPID=0、端口已消失；空辅助scope已停止。
- 远端缓存复用 D78/D80，不重新发布；60份真实交付记录和远端监控已取回，
  SHA 与远端一致。原始请求、worker、resolver、物理计量记录全部保留。
- 保护清单147项前后相同，验证 JSON SHA 均为
  `96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d`；
  唯一历史差异仍是用户 replay 修改，未重置或暂存。
- 离线分析15项CPU测试通过（0.050秒）；D409配置扩展对应基线89项测试
  通过（7.234秒）；均为4GiB/零swap辅助资源域。不是新GPU性能实验。
- 本轮没有改论文、阈值、权重或trace，没有生成图；全7B比较、Resident预算、
  W1/W2/G2、3B、A1–A5、S1–S13和最终论文图表仍未完成。
