# D413：ServerlessLLM 测量入口连接生命周期验证

日期：2026-10-11。**CPU 验证通过；尚未进行新的 GPU 性能实验。**
完整技术记录位于 baseline 仓库
`docs/ieee_tc/D413_INGRESS_CONNECTION_LIFETIME.md`。本记录不替代完整计划。
Baseline 检查点 `35dda117faecee5375c146a2c23b16fc2562d82a` 已推送
`origin/main`，并核对远端分支 SHA 一致。只提交了本次两个脚本、一份测试
与一份说明；既有用户 replay/relayserve 修改未纳入。

## 对论文证据的影响

D411 的 4,000 请求完整回放保留：3,997 个响应通过原生计量/身份校验，
3 个请求在额外测量入口向本地原生 API 转发时失败。因此即使联合 SLO 为
98.45%，也不满足全部正确完成的前提，不能进入 G1/G2 排名。
这些失败不是远程 LoRA 工件传输失败，不能据此归因于 ServerlessLLM 的
调度、扩缩容或加载机制。没有删除失败，也没有改写旧结果。

三次失败距更早响应结束均约 5 秒，和实际 uvicorn 的空闲关闭时点一致。
原日志没有连接 ID，所以该关联不是历史根因的唯一证明。对实际入口做了
事件顺序可控的真实 HTTP 连接测试：选中复用连接后关闭对端，第二请求
返回一次 502，且未到达原生服务。该测试证明故障机制可发生，不估计发生率。

## 修正及结果

只让这段额外入口使用“一请求一条本地连接”；不改 ServerlessLLM 的原生
策略、vLLM 配置、SLO、准备窗口、工作负载或物理计费，不增加自动重试。
连接建立时间及其服务资源消耗照常计入，没有事后扣除。该规则是测量适配，
不是论文机制贡献，也不构成新的性能调参。

| 验证 | 结果 |
|---|---|
| 修改前真实连接顺序验证 | 第二请求一次可见失败，未重试；1 测试通过 |
| 相同关闭条件下的新连接方式 | 两请求成功、不同连接、无复用 |
| 服务已收到请求后连接中断 | 保留一次失败，不重复执行推理 |
| 到达/并发/取消/超时/原生错误检查 | 8 项入口测试通过，0.345 秒 |
| 既有计量、启动和调度回归检查 | 92 项通过，7.538 秒；4 GiB、零 swap CPU 范围 |

依据实际 aiohttp 3.13.3 和 uvicorn 0.41.0 的固定版本源码：
[连接生命周期](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/connector.py)、
[POST 重试边界](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/client.py)、
[服务端空闲关闭](https://raw.githubusercontent.com/encode/uvicorn/0.41.0/uvicorn/protocols/http/httptools_impl.py)。
没有用人为缩短超时或自动重试掩盖失败。

历史 helper 可从 baseline 执行 commit
`2a5dfd1154ba93c82d9b7076fde0a8521543933b` 精确恢复，其 SHA
`db0c5232692215a1dd3b8ded3787b7f0dd6dc546e5e8a33bdfc9fbaed5353740`
与 D411 原 manifest 一致。当前工作文件不是旧运行源码。

## 下一步与未完成事项

备份这次已验证修改，核查并仅清理 D411 可重建下载对象，恢复推理机至少
150 GiB 的新重型运行门槛。随后用新的测量版本身份进行一次完整 W0 回放；
D411 原失败全部保留。复用已通过的全池/后端验证，不为这个连接修正重跑
100、500、1,000 请求 GPU 检查，不改变已有公开配置或寻找更有利重复。
新回放结束后完整审计、封存，再推进 7B vLLM。

PrimeLoRA D371/D397 继续封存。7B 完整 baseline 比较、有效 Resident 预算、
W1/W2/G2、3B、核心消融、机制与敏感性均未完成；不能把本次故障验证当成
这些性能实验已经完成。尚无新性能值、优胜或改善幅度结论。

本次使用 run-experiment 限定既有测试入口与资源范围，使用 analyze-results
将观测、机制解释和证据边界分开；不生成图、不增加运行间 CI。
