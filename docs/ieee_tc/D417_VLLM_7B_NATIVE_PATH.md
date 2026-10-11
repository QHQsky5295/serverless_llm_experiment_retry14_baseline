# D417：vLLM 7B 原生测量入口的受影响路径资格

2026-10-11，启动前登记；15:33:21 启动，约15:35终止；启动环境错误，已收尾。
主仓库执行版本 `cb2561938bc7871a2b0cfb23117417223a92d25a` 已推送并核验。

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

启动前编译缓存 221,763 文件、7,506,097,394 bytes，聚合 SHA
`42cdcdbff103f9cf09cd860d9411bb6071f588087d4837eb84d40d386160ce29`。
仅扫描既有磁盘缓存，不构造 engine、不预载 GPU；冷 LoRA cache 单独创建。
远端实际链路 1 Gbps/full。四个原生 EngineCore 的实际服务资源域和 CPU 范围
已由 watchdog 读回；初始无 high/max/OOM/swap 或保护中止。完整运行身份、
原输入/配置 SHA 和远端 InvocationID 已保存于 EXECUTION_STATUS 与原始回执。

## 终态：启动工具查找失败，不是性能点

四个 worker 均在 FlashInfer sampling warmup 报找不到 `ninja`。工具实际已安装于
声明的 venv/bin；D416 新入口调用绝对 Python，但没有传播对应 bin 的 PATH。
历史 D214retry6/7 已出现并处理同一要求，这次属于入口集成遗漏。
原生图捕获、模型加载已执行，但没有请求生成结果；不能归因于 vLLM 吞吐、
远端制品或其调度机制。原请求、公式、后端配置均不据此修改。

- 100 条输入合同；最终 27 个请求被创建、24 个已发送头、27 个失败，0 个响应。
  其余尚未到达的 73 个不填成 timeout。
- 物理记录较早截止于26个终态、4个未合格关闭的 owner；保留
  `U_obs=499.79549817810766 GPU-s`、`U=null`，不进入成本/SLO排名。
  外置清理已证实真实 GPU 释放，不能据此追改原 owner 日志使其“合格”。
- 服务内存峰值21,910,249,472 bytes；主机可用内存最低102,155,436,032 bytes；
  无 high/max/OOM/swap 或保护中止。147个历史保护条目零变化。
- 三个远端服务均 inactive/MainPID0、端口无监听，空辅助域已停止。
  原 stop 脚本在停止成功后寻找不存在的传输日志时退出1；单独复核成功。
  远端监控已复制并核验SHA；不存在推理响应或本次制品获取证据。

审计 `paper_results/ieee_tc/baseline_audit/d417_7b_startup_failure/` 保存审计和
42文件证据清单；原始 `close_failure.py` 是一次性离线收尾配方，不修改测量。
下一步 D418 修复声明环境的工具链继承，CPU验证后登记新执行键。
不原样重试D417、不关闭FlashInfer、不把失败归入性能对照。
