# D418：vLLM 启动环境修复与 D419 前置验证

2026-10-11。D417 已终止、资源释放并保存失败；不是性能结果。
回查历史 D214retry6/7，旧入口已明确将 venv/bin 加入 PATH；D416 新入口遗漏。

本轮在既有 runner 中绑定声明的 Python 与其工具目录，GPU 分配前验证
实际 `ninja` 可执行文件与版本，各 worker 再核验继承值。缺失或外来工具
显式报错，不安装新环境、不关闭 FlashInfer、不自动切换后端。
原计划的构建保护以显式 `build_max_jobs=2` 传给原生 MAX_JOBS；并发同一
sampling 模块的构建使用 FlashInfer 自带文件锁。该设置进入新执行身份，
不改变推理策略和公共 engine 参数；启动/JIT 开销仍计入真实生命周期。

依据：已执行的 D417 日志、D214 历史；
[官方 FlashInfer 调用](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/cpp_ext.py)、
[原生锁](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/core.py)、
[Python venv](https://docs.python.org/3.12/library/venv.html)。
本地 flashinfer-python 为0.6.18.post1；引用tag支持调用约定，不宣称整文件相同。

125项 CPU 测试通过，7.756秒；4GiB/零swap/既定辅助CPU。
实际 venv 隐藏GPU探针确认 ninja 已可调用；提取安装源码的两个原生方法
验证命令含 `-j 2`（构建命令为mock，不启动编译）。原生二进制SHA与探针
回执位于 D419 的独占准备目录。测试不等于 GPU/数值正确性资格。

下一步仅做原100请求受影响路径 D419，随后审计复用资格并推进7B完整W0。
PrimeLoRA D371、ServerlessLLM D414保留封存；U_ref、W1/W2/G2仍待完成。
详细代码与测试说明在 baseline 仓库同编号文档。

已测试 baseline 版本 `3bd6088ce988a4108f397378c5b59a8e586cbc8f`
已推送 origin/main 并核验远端SHA。只提交四个任务文件，两个用户修改的
baseline脚本SHA不变；diff/secrets检查通过。
