# D434 — S-LoRA 7B original1000 compatibility

2026-10-11 preregistration; launched22:25:08Asia/Shanghai, no terminal outcome yet.

Execution MAIN47be5422cbf65c8c0905f78668a83c188b7b9ccf,
BASE5059cb0b86c5a5378a772adaa25aa80bb7a1b240; both pushed/remoteSHA verified.
Tmux tc-d434-7b; service InvocationID18c0f4b4575a45be824a81b181b46476.
Current source/runtime configuration unchanged; monitor this same run, no repeats.

## Question and fixed scope

After D432 native100 and D433 all500-ID functionality, execute the missing original
1000-request compatibility view once. Reuse original seed42 trace, first1000 entries,
60 requested logical adapters,122510 target output tokens. No remapping, regenerated
trace/weights or baseline performance tuning. This is qualification/development,
not full4000 W0, a baseline optimum, G1/G2 ranking or independent holdout evidence.

Native config remains slora_7b_resident_native_v1.json SHA
2dd3a7415c5b00f0bbcf6ac7b31906da4399285e6df824ba270988015c72ae5d.
DP4/TP1, original ordinal round-robin, sequential worker startup, native S-LoRA
paging/swap/scheduling, D428 packed HOST views and D430 measurement/source view.
Full500 static HOST preparation starts after notice; business starts60s after notice
without waiting for ready. Actual native allocations and release define GPU-s.
No changes to baseline runtime, scheduler, parameter values or qualification thresholds.

## Input, resource and remote contract

View SHA 0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981,
source SHA efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c.
Same fixed_length_greedy_v1/canonical prompts/native SSE token IDs.
Frozen D394 common SLO, frozen metric V1 + single-run V2; new baseline execution rules.
Independent numeric limitations remain; raw unknown fields are not promoted.
The original1000 view does not reuse D433's adapter-remapped performance numbers.

Exclusive raw root:
 /home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d434_20261011
Private root: /tmp/tc-d434-7bs. Reuse D433 orchestration, changing paths and population
only. No previously successful100/500 gate repeated. Existing Triton disk cache is
sealed separately; no prestarted GPU engine or free adapter preload.

One heavy job; service72/80GiB,swap2GiB,CPUs4-23,28-47; auxiliary3/4GiB,zero swap,
CPUs2,3,26,27; independent watchdog and preflight disk150/100GiB rules.
Reuse D78/D80 readonly published gzip cache. User reapproved once-only cache;
do not republish or duplicate extracted pool. Actual remote reads/transmission/
contention and local materialization remain measured; request-time packing absent.
No remote changes, hashing/cleanup/tests or concurrent heavy work during service.

## Acceptance and continuation

All1000 offered/submitted/terminal/native/source/adapter/time contracts, no fallback,
actual_tokens==target for every request; complete original-view identity.
Physical actual-worker leases closed; mandatory timing recomputation within1ms.
Full500 deployment transfers count once, not once per referring request.
Report common joint SLO even if below95%; that does not trigger baseline tuning.
Use existing audit_d402_serverlessllm_path.py with scheduler slora_resident_rr_v1,
population prefix1000 and common D394 manifest. Do not apply prefix100-only numerical
reference argument to this full1000 population or invent a new GPU oracle.

After actual GPU release: close only owned remote units, preserve receipts/journals,
audit/protect/seal and back up. Then safely reclaim only proven recreatable private
downloads if needed; fresh gates before one full4000 W0 using the same configuration.
PrimeLoRA/ServerlessLLM/vLLM sealed W0 points stay unchanged;3B and later fullmatrix
remain open. No figures or paper edits in this qualification step.

Prelaunch CPU checks:37 passed1.858s with existing Python3.12 environment.
Initial systemPython3.10 attempt had a missing numpy import; failure retained,
correct interpreter reused, no dependency installation or runtime modification.
Fresh resource preflight passed;147 protected historical entries unchanged.

## 终态与证据核验（2026-10-11）

22:44:35前回放结束、实际GPU已释放。1000 offered/submitted/terminal/native/source/
adapter/计时合同全部通过，零失败，122510原生输出tokens，60实际逻辑adapter IDs。
四个声明TP1副本均实际服务，各原生500-ID注册表与来源/配置/真实物理持有者一致。
E2E、TPOT、native计时重算最大误差均0ms。此前D432/D433资格不重复。

| 原1000兼容验证的描述性观测，非完整W0 | 数值 |
|---|---:|
| 平均 / P95 TTFT | 37636.062 / 236234.733 ms |
| 平均 / P95 TPOT | 33.501 / 49.954 ms |
| 平均 E2E | 41782.526 ms |
| TTFT / TPOT / 联合SLO通过 | 750 / 999 / 749（分母1000） |
| 物理GPU生命周期 | 3542.956035 GPU-s |
| GPU-s / binding-valid请求 | 3.542956 |
| 服务内存峰值 | 62617686016 B |
| high / max / OOM / swap | 全0 |

共同联合SLO74.9%，不满足95%。本轮是资格/开发前缀，不作为完整W0主比较、
资源最优点或预算参考；不因未达标调参或重复同配置。完整4000仍按既定顺序执行。
原始n_correct=null、独立数值资格/全系统资格未知或false保持不变；沿用D432
已记录的有界独立见证范围，不增加GPU oracle，不将全输入绑定等同全数值证明。

### 阶段解释与边界

- 500静态工件准备跨度95.131748s，在共同通知之后，CPU-only段不虚构GPU-s。
- 四个原生worker全模型/全池就绪相对通知分别221.661525、320.676475、
  421.198832、518.431829s；共同业务仍在通知后60s开始，未移动到ready之后。
- 全体平均router queue32817.474ms、service TTFT4802.203ms。
- 原序号0–99：联合1/100，平均TTFT210244.181ms；
  100–499：249/400，平均41227.250ms；
  500–999：499/500，平均241.488ms、P95 371.496ms。
- 这支持“启动/积压阶段与后续稳定期差异明显”的描述，不是新因果消融。
  不把后500当独立实验或启动免费的主结果，不把全体时延都归为网络。
- 全500部署传输共420740410线上字节，12985984450验证物化字节，0请求打包。
  1000请求引用这批准备，不重复计为1000次新传输；请求线上新增字节0。
- GPU互斥窗口：pre-arrival0、arrival3493.010659578、
  drain23.431281582、cleanup26.514093614，合计3542.956034773 GPU-s。
  原生真实退出/释放已验证，CPU-only准备没有虚构持卡。

沿用既有严格slora_resident_rr_v1/prefix1000 analyzer，无新运行时或统计补丁。
审计SHA 05a595c69063efc92484a208325b3476fd4c49c857a6a450b281845af39660a0。
578文件/63678229B证据清单SHA
bf444b0c1ce339e6a63342b6484e8cc2dd1d15fbe2d317163d9a9378c65ff7ab。
来源包含原生请求、进程/资源记录、双端传输、内容收据、配置与执行身份。
主机最小可用60923981824B，磁盘最小150234251264B，无运行中保护中止。
147历史保护项零变化，三个用户修改文件和原S-LoRA十文件补丁SHA保持不变。
本轮三个远端服务按InvocationID停止、MainPID0且端口无监听；未重新发布缓存。

下一步：核验清单并备份，按原引用检查仅回收可重建私有下载副本；
fresh安全门槛通过后，使用完全相同S-LoRA配置运行一次完整4000 W0。
不重跑100/500/1000，不优化baseline性能；保留PrimeLoRA、ServerlessLLM、
vLLM既有封存点，不跳到3B。全目标与未完成实验矩阵仍保持OPEN。
