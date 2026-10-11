# D417：vLLM 7B 原生测量入口的受影响路径资格

2026-10-11，启动前登记。状态：准备中，尚未启动 GPU；后续按执行记录更新。

本轮不是新参数搜索。D416 接入准备期独立到达、服务侧远端获取、原生 token
时间和真实物理所有权后，需要一次实际 GPU 验证。D395/D396 的历史结果
继续保留为 R2，不用于 U_ref；PrimeLoRA D371、ServerlessLLM D414 均不重跑。

## 冻结本轮身份

- baseline 代码 `5c0cfd8b4d4ead40437d6dc852a9e60395075cf3` 已推送并核验；
  配置 `configs/ieee_tc/vllm_7b_resident_native_v1.json`。
- vLLM0.30.0，四个 TP1 原生 AsyncLLM；原有 RR，maxseqs2、GPU LoRA4、
  CPU LoRA16、GPU fraction0.7、FP16、chunked prefill、无 prefix cache、非 eager。
  原生磁盘编译缓存允许保留，启动前记录身份；GPU/HOST/KV/本轮 LoRA cache 冷。
- 仅复用原 7B W0 前 100 请求，不改 adapter/arrival/prompt/target。
  源 trace SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。
  是受影响路径功能资格，不是完整 4,000 请求性能点或 500-ID 覆盖。
- fixed_length_greedy_v1、role_lines_v1、原生 integer token IDs；共同 60 秒
  准备与 1,800 秒资格保护；D394 共同 SLO 不变，不以短前缀判系统最优。
- 实际 HTTP 访问既有 D78/D80 已发布只读缓存，禁止请求中打包、模拟等待、
  本地权重 fallback 或整池复制；请求触发的必要物化、传输与竞争计入。
- 72/80 GiB 服务、swap2 GiB；4 GiB/零 swap 辅助域；既定 CPU 集合，独立
  watchdog。启动磁盘至少150 GiB、主机可用内存102 GiB，预估新增峰值8 GiB。

## 原始目录与验收

`/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d417_20261011`

由已有 common launcher 启动现有 vLLM runner 的显式服务模式；使用新私有
缓存 `/tmp/tc-d417-7bv`。预置数据不重新生成；所有旧结果和用户修改保护。

通过条件：100/100 原生请求合同，首末 token 与时间分解可靠，实际远端内容
与交付收据对应，服务/辅助归属可读回，四卡 allocation/worker/pidfd/release
证据完整。失败与未知数值正确性不改写；没有完整资格就不冻结预算参考。

结束顺序：释放 GPU → 按 InvocationID 停止本轮远端服务 → 保存 remote/
resolver/worker/资源证据 → 校验、分析、备份。业务中不改远端、不清缓存、
不扫描大文件、不跑其他重型任务。随后审计可复用的后端资格并进入完整 W0。
