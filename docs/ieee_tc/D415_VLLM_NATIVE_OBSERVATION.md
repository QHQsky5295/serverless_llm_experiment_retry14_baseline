# D415：vLLM 原生计时组件已验证，完整入口仍待接入

2026-10-11。接续 D412/D414，当前主线为 7B vLLM；不是新性能结果。

## 已完成

在 baseline 仓库现有 `scripts/ieee_tc_serverless_measurement.py` 中增加
显式 vLLM 0.30.0 原生快照入口，复用 `NativeV1TokenTimeline`，不引入
PrimeLoRA 控制机制。不改用户 dirty replay，不改已安装后端、输入或权重。

核查了历史 `26b1628` 观测代码和 D395/D396/D397/D412 的计量结论，并在线
对照官方 v0.30.0 的 output_processor.py、outputs.py、stats.py。
三个官方源码的 SHA 均与现有安装相同。实际原生方法的 CPU 测试复现可变
统计/token 列表引用以及 collector 不传播 metrics 的行为；新增快照使原生
token 数与同事件时间同步，完成通知不移动最后 token 边界。

新增 13 项原生组件检查及既有回归共 **112 项通过，7.803 秒**，
使用 4GiB/零 swap/固定辅助 CPU scope。实际 0.30.0 环境的两次幂等
安装也通过；未启动引擎，GPU 计算进程前后为空。
首轮测试曾有一个异常传播预期错误，已在 baseline 报告保留，不改原生异常语义。

详见 baseline 仓库
`docs/ieee_tc/D415_VLLM_NATIVE_OBSERVATION.md`：源码 SHA、官方链接、
改动边界、测试清单与未完成事项均在其中。

## 必须保留的证据范围

这只是观测组件，不是完整服务入口、更不是新 Resident 参考。
尚需接入服务域内 ordinary routing/真实远端按需获取、部署前接收入口、
独立 open-loop 到达、真实物理 lease/worker 释放以及端到端逐请求关联。
不能原样重跑 D395/D396，不能把 SSE 首字节或 EOF 当原生 token 时刻。
不改变 G1/G2、D394 SLO、单轮规则、公共配置或 60 秒准备期。

磁盘约 146.9GiB，仍低于新重型任务 150GiB 启动底线；未清理 D414
下载 objects，未启动 GPU 资格或远端服务。下一步完成共同入口，再核验并
仅回收已结束且可重建的缓存，重新通过资源门槛后运行受影响路径资格。

PrimeLoRA D371 和 ServerlessLLM D414 继续封存；没有新排名、最优性、
3B、W1/W2/G2、消融或敏感性结论。远端一次性发布缓存已完成 D78/D80，
用户本次再次批准不产生重复发布或解压池。

## 备份与保护

Baseline 测试节点 `aba520572d019743320416b0015b1ba701716ee0` 已推送至
`origin/main` 并核对远端 SHA。只提交本任务的观测代码、测试和说明，
用户已有修改未 stage。旧结果/图表/用户文件 147 项相对 D414 前置清单
零变化，回执位于 `paper_results/ieee_tc/baseline_audit/d415_vllm_native/protected_after.json`，
SHA `6b4deabad0cf41b4f928ab8d0069edd8ae445128fb75786a9739f4e9baf11a70`。
