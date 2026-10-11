# D425：S-LoRA 7B 历史复用与计量入口核查

日期：2026-10-11。性质：离线来源、执行合同和已有 CPU 回归核查；不是新的
GPU 实验、性能排名或独立完整语义审计。7B 主线继续，PrimeLoRA D371/D397、
ServerlessLLM D414 和 vLLM D423 不重跑。D424 的比较和 W0 Resident 参考保留。

## 结论

已有 S-LoRA 环境、500-ID 工件集合、固定输出和 FIFO 修正可以继续使用。
所查历史记录不能直接补成当前 G1/G2 主结果：完整回放没有原生输出计数合同，
固定输出检查没有原生首末 token 时刻和物理 allocation/release 证据；旧部署
在全部副本 ready 后才开始回放，也不同于本轮统一提前 60 秒通知。

因此不重建 S-LoRA，不改它的调度、paging、batching 或 adapter 策略；下一步
只接入原生观测、统一启动和物理资源记录，再验证受影响路径。旧数字全部保留，
不能通过从 E2E 减去估计开销或把 startup 最大值当持卡时间完成“离线修复”。

## 1. 实际查到的记录

复用 D420 inventory 的逐 JSON 逻辑，仅调整 S-LoRA 选择条件并添加配套日志、
配置和源码关联。选择器：BASE 下 `rg --files results -g '*slora*replay.json'`，
路径包含 `7b`；共 25 文件。它不表示对服务器所有可能命名的文件作穷尽证明。

| 历史组 | 运行数 | 每轮请求 / reported success | 实际逻辑 ID | 原生计数 / 固定长度匹配 |
|---|---:|---|---|---|
| 4–5 月主表、负载、pool、local-sim、真实远端 |17|4000 / 4000|44–132|0 / 未记录|
| 7 月 18 日首次 fixed-output |1|100 / 100|29|100 / 85|
| 7 月 18 日 FIFO 定向检查 |1|32 / 32|13|32 / 32|
| 7 月 22 日固定输出检查（含 campaign attempt） |3|100 / 100|29|100 / 100|
| 旧 fix3、fix4、fix5 开发回放 |3|500 / 395、396、500|86|0 / 未记录|

`reported success` 只是原始标志，不升级为冻结协议的正确完成。三份 7 月 22 日
记录全部保留，不按性能挑选一份；两个失败开发回放同样保留。未找到实际执行
500 个不同 ID 的记录：名义池为 500 或初始化加载全池，不等于 500-ID 请求覆盖。

25 份回放的 `server_metrics` 均为空；原生整数计数只出现在五份 fixed-output
记录中。现有请求字段无后端首末 token 时刻、物理 owner 或释放区间。另关联
230 个支持文件，其中 107 个日志共 486,996 行；登记的原生时间/lease/resource
关键字没有匹配。关键字未命中单独不证明证据绝对不存在，本结论还结合了下面的
源码和完整代表日志审阅。详细路径、字段、SHA、字节数保存在 inventory。

## 2. 已完成的源码与历史关联

- 上游 `c1ddf488781ea7f551cd0bb07bfd097124c93411`；仍使用已有
  `slora_official_cu118`，不升级、重装或覆盖该环境。
- 当前 10 个既有 tracked 兼容修改与保存的 patch 解压内容逐字相等。
  普通 binary diff SHA 为
  `7a0f48f60c225789014ad960ec3ab1cf6388611b7650a4f656f1208c981c2d60`；
  `--full-index` diff SHA 为
  `c62bb51e7ec316e19b60f68807b7eba0fdfe17cdd8dfe67ab7147807a3d64b89`。
  这是相同文件的两种 diff 格式，不是两套性能实现。
- 三份 7 月 22 日 sidecar 的 10 个修改文件 SHA 全部与当前一致。7 月 18 日
  记录不能混称同源码：首次少了 HTTP FIFO 修改，另有 embedding-loader 字节差别。
- 当前 `ModelRpcServer.forward` 的 AST 与固定上游版本完全相同；这只证明该
  方法未变，不证明所有 kernel 或数值行为已经通过独立验证。
- 已经通过实际 HTTPS 获取核对 `api_server.py`：上游响应、Git HEAD、本地
  文件三者逐字相同，SHA
  `504a1128cd8c6868a7d4c4ed21127b93f3cc7db0fc009082677bf1c1de97c485`。
  这补上了上一轮网页读取失败留下的来源缺口。

历史修改依据同时核对 BASE 的 `75d29eb`（FIFO 文档/测试）、`1597ebe`
（formal gates）、`a98ee47`（fixed-output 原生计数）及现存完整 patch。
没有重置或修改嵌套工作树，也没有修改用户的 replay/relayserve 文件。

## 3. 为什么必须补计量路径

原生 `forward` 在采样结果搬到 CPU 后才构造 token metadata，目前只含 ID 和
logprob。router、detokenizer、FIFO 传递该 metadata，但 `generate_stream`
重新选择 SSE 字段。因此，旧客户端所见 chunk 时间不是可恢复的原生 token
产生时间；FIFO 修复解决事件丢失，不自动解决事件计时。
依据：[固定版本 model_rpc.py](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/model_rpc.py)、
[固定版本 API](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/api_server.py)。

旧 wrapper 按顺序启动四副本并逐个等待 ready。5 月真实远端运行日志记录
141.111049、153.101674、149.515772、187.051238 秒，合计约 630.780 秒；
fleet 却只保存最大 startup 187.051238 秒。该和只是四段已记录准备耗时，
不是物理 GPU-s，也不含所有前置阶段。7 月 22 日 15:41 检查同样为约
571.201 秒合计、168.282 秒最大值，再开始业务。这不能满足共同通知时间线。

官方实现会在主存保留全池、按运行请求在 GPU 准备 adapter；这一原生行为
应保留，不能强改成 PrimeLoRA 式按需层级策略，也不能把本机准备移到通知前。
依据：[S-LoRA 官方项目说明](https://github.com/S-LoRA/S-LoRA)。
CPU-only 下载阶段不虚构 GPU 占用；真正持卡后持续计至实际释放。

## 4. 最小实施边界与下一入口

1. 在现有通用计量组件增加显式 S-LoRA 事件合同：在实际采样 token 可用后、
   detokenization/HTTP 之前记录本机 monotonic 时间，绑定原生 request、batch
   adapter 路径、输入 token 和实际进程。completion 尾部单独记录，不并入 TPOT。
2. 保留上述 10-file 兼容版本。新增观测必须显式 opt-in、固定源 SHA，优先用
   独立受控 source view；不覆盖原树，不复制模型/工件，不偷偷改变 sampling、
   scheduling、paging、load/offload 或全池 HOST 准备。
3. 复用现有共同 launcher、pre-ready HTTP publication、published-artifact
   resolver 和物理 lease/实际释放；只增加明确的 S-LoRA 后端分支，不把它
   重命名成 vLLM 或 ServerlessLLM native-store 来绕过验证。旧 shell runner
   的端口清理、旧目录删除和 ready 后才回放路径不能直接用于本轮。
4. 先 CPU 合同测试，再一个原始 100 请求受影响路径检查；复用既有生成/FIFO
   证据，不为逐 token 完全相同反复运行。当前 500-ID 执行和 1,000 请求新合同
   兼容证据缺失，按计划完成必要资格后运行一次完整 4,000 W0。
5. 真实远端复用 D78/D80 已发布只读缓存，不重新发布。统一初态、提前 60 秒
   通知、共同 D394 SLO 和完整物理资源合同不变。对性能好坏不设准入筛选。

上述观测/启动适配尚未实施或 GPU 验证；D425 只关闭历史复用判断，不关闭
S-LoRA 资格或性能任务，也不开始 3B、消融或敏感性。

## 5. 验证与产物

- 已有 fixed-length/FIFO CPU 回归 9 项通过，0.073 秒；4 GiB、零 swap、
  CPU 2/3/26/27，GPU 隐藏。原生计算正确性/500-ID GPU 执行未因此推定通过。
- 276 个来源引用逐个 SHA/字节核对；旧保护清单 147 项零变化。
- inventory：`paper_results/ieee_tc/baseline_audit/d425_slora_reuse_inventory.json`，
  SHA `95f471cb65cf5646b5a545221dceae5afdcb3f1fddf341582b5b014e70f71161`。
- source evidence：同目录 `d425_slora_source_evidence.json`，
  SHA `bb673cafeae78ba5f3de5c901a4477ff9147c2521132079c3722f7f9b80420c4`。
- raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d425_20261011`，
  含复用的清点 recipe、来源核对 recipe 和真实 CPU 测试日志。未生成性能图。
- 小型复核包：同 curated 目录 `d425_slora_reuse_recipes.tar.gz`，仅含上述两个
  recipe 和 CPU 日志；SHA
  `8bd8b22872e57ea6fce41f45d1aaaee0c5a1bd3c7604b21fb20f2d30a7ddcf44`。
  六个本轮辅助资源域均在 InvocationID、限制和进程为空核对后关闭；未终止其他进程。

采用 analyze-results 的“观察—解释—边界—下一步”流程。没有新增性能数据、
更改评价阈值、重建权重/trace、清理旧证据或远端操作。
