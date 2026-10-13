# D486：HydraServe 原生编译完成与实际加载验证

2026-10-13。主线仍是 HydraServe 实际结果 → 全部可用基线比较 →
PrimeLoRA 优化。没有新性能运行，没有改变任何基线的调度/扩缩容参数。
原有 k3s 不属本项目，未访问或修改。

## 已完成的原生构建

D485 header_retry 的同一编译进程已正常完成29/29步骤，elapsed
2198.173899430083秒；没有重新启动或换掉原生后端。fastload复用首轮
已成功产物。源码、CUDA工具链的固定文件SHA前后核验通过，官方checkout
`8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`仍干净。

| 原生库 | 大小（B） | SHA-256 |
|---|---:|---|
| _C | 62764912 | 3cde8d34b0f6f285bf742cd2bb14f75ed653941a7b6932268d42b23f782e5c0a |
| _moe_C | 11332880 | 9fb0c97f5fdb4d730fb63c076ff5f3f45a869c5f9997ae71449446e553d7218d |
| _punica_C | 45215848 | 05c9376e8c93b02307d7ba2e39960c2cf2b41c23b81eac9a8cd4e6d0945e0214 |
| fastload | 12279432 | f11b31a82f654d7c8950d85e05461d31b5cc45469720deb1e784d47022571e44 |

实际递归成员为空后按原InvocationID/inode关闭scope。峰值3221487616B，
memory.high557、max/OOM/OOMkill/swap为0；不将high写成0。
完成时磁盘可用203100196864B。编译/链接不等于CUDA、LoRA或性能资格。

## 依赖与打包方式

继续同一个D484独立Python3.10/Torch2.3.0/xformers0.0.26.post1环境。
不向旧环境安装或升级，不修改共享NVIDIA运行库，不重新下载模型/LoRA/trace。
利用固定官方setup.py已经提供的 `VLLM_USE_PRECOMPILED`：将逐文件SHA相同的
Python包与本次三个真实扩展放入小型私有source view，按原setup安装，
不重编已成功的内核，不以普通PyPI vLLM替换。

FlashAttention选官方2.5.8对应的CUDA12/Torch2.3/cp310/旧C++ABI预编译包。
选取依据是已测ABI和[该版本官方打包规则](https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.5.8/setup.py)，
不是性能版本搜索。[官方发布文件列表](https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.5.8)
包含该文件；实际安装仍须整包来源/记录和动态导入检查。
GitHub API本轮403限流，未不断重试；官方发布页可查，实际HEAD200，
Content-Length120922965B。HEAD成功不是已安装成功。

NCCL保持原requirements的2.18系列，选择官方vllm-nccl-cu12 2.18.1.0.4.0。
已读[PyPI源包](https://pypi.org/project/vllm-nccl-cu12/2.18.1.0.4.0/)的setup.py：
它会循环下载并写用户全局~/.config，故在独立private_home中先有限下载、
核对其官方MD5，再运行未修改的源包安装；父级1800秒保护保留。
库来源为[vLLM官方NCCL发布](https://github.com/vllm-project/vllm-nccl/releases/tag/v0.1.0)，
HEAD200/291649520B；不能把Torch自带2.20.5默认为所需2.18.1。
记录下载后的完整SHA，并将私有文件权限收紧回0644。不输出任何凭据。

实际加载检查先隐藏CUDA，只验证原生Python/动态库及版本一致性；fastload
此步使用ENABLE_PARA=0以避免其静态初始化意外创建共享内存/加载模型。
此设置只用于导入检查，不是正式服务关闭原生加载机制的配置。
真实GPU、共享内存加载、非零LoRA及迁移/合并资格仍必须单独完成。

## 当前边界

原始目录为实验仓库hydraserve_qualification/d486_20261013；复用D485有限
子进程监督器、资源准入和关闭方法。全部依赖/原生包安装及pip check实际
通过；原生vLLM0.4.2（distribution0.4.2+cu124）、三个扩展、FlashAttention2.5.8、
fastload和NCCL2.18.1实际导入通过，CUDA未初始化。仍不是GPU执行或性能资格。

第一轮80.083247秒，最后清单核查失败：检查错误地要求安装源码中的
fused_moe/configs/README，而原setup的package_data只包含该目录JSON。
全清单核对确认只有这份说明文件未安装；不是缺内核/Python实现或运行数据。
保留原失败、源码与日志，仅新建检查入口排除这一确切README，所有其余
Python、JSON、py.typed、扩展仍逐文件SHA核验。不重装、不重新下载或编译。
修正后3.184380秒实际通过；原生文件没有因核查而修改。

两个任务均实际递归空、按InvocationID/inode关闭。第一轮峰值1482092544B，
第二轮259051520B；high/max/OOM/swap全0。3/4GiB、swap0、CPU2,3,26,27，
完成时free202032189440B。NCCL库完整SHA为
`005503b62b9fe466b6b2f60ba2ad1daa4571196ffcb82a3d77a3c775bfcd5ac8`。
原pip报告保存实际下载来源/SHA。无GPU/model/远端任务；没有重新生成数据。

同轮只读D414请求级SLO复核通过；它是既有ServerlessLLM数据分析，不是
HydraServe性能结果。解释文档位于主仓库D486_METRIC_FAIRNESS_RECHECK.md。

后续仍需将D479层映射/共享owner、D480平台接口、D481私有IPC/地址适配
接入实际原生worker，完成非零LoRA、原生固定输出/计时和100→500-ID→
1000→4000。不触发的pipeline/consolidation如实记录；单机四卡不冒充多节点。
7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13及最终文档全部保留。
