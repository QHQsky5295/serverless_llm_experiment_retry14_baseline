# D519：恢复已封存服务语义后的单次迭代诊断

2026-10-14。承接 D516 的未达标结果与 D518 的源码/CPU 观测资格。
本记录是运行前登记，不是性能结果。仅一次原始1000请求前缀，不新增负载。

## 恢复与不变条件

D514 将 GPU preparation 执行前 source_snapshot 改为 source_identity_snapshot。
D516 完整4000请求观测联合 SLO 94.775%，未满足冻结95%，因此不接受该候选。
这不是单轮因果退化证明。D519 恢复这一行，主运行脚本与 D476 生产源码
db44e8f068eb30c69466f32ad3d147ffbc45f9e2 逐字节相等；四项新增 GPU 所有权
回归测试保留，改为检查已恢复的完整 source_snapshot 入口。HOST 的 D474
优化保留。D514/D515/D516 的源码、失败及结果均不覆盖。

faaslora 相对 D476 仅包含 D518 的两个 opt-in 观测文件差异，不改变默认服务。
复用 D512 runner/原始1000索引、D476配置和D78/D80只读交付；三个结果及缓存
目录使用独占新路径。九公式、路由、admission、allocator、缓存额度、模型、
SLO、生成与输出上限不变。无未来信息、无新增权重、无新 Kubernetes 操作。

## 可证伪问题与测量

历史 D517 单token分配迭代均值22.71ms，原生 scheduler 约0.22ms；多token
迭代约80.71ms。不能从这些外层跨度推定 GPU 计算或锁页分配是瓶颈。
D518 源码表明零大小锁页缓存阈值作用于整个进程，V2输入准备也使用锁页缓冲。
本次测量检验其驱动分配时间是否足以解释主要步骤耗时，而不是直接改变阈值。

复用 host_spans_v1，明确启用 v2_components_v1：实例内方法次数、主机 wall/
线程CPU、进程分配/释放计数与微秒累计时间。只读计数；无新 CUDA event、
额外同步、Future替换或请求内容记录。嵌套不可相加、后台释放不可直接归因
单请求、主机 graph 调用不等于 GPU kernel 时间。详细观测开销不从结果中扣除。

运行资格要求原生1000/1000完成、输入/输出合同、远端及真实释放可核查；各
进程观测首尾完整、未达64MiB/200000行上限、step字段完整、计数无倒退。
若失败/截断，保留实际数据和缺口，不冒称诊断成功；不重复同键筛选结果。
本次只作诊断，不参与主表或 G1/G2 排名，也不把原生数量证明当独立数值证明。

## 顺序与安全

CPU回归/源码及保护清单 → 备份 → 实际CPU资源域清空 → 远端启动及认证health
明确返回 → 新鲜磁盘/内存/GPU门槛 → tmux启动一次。运行期间仅轻量状态
监控，不测试、备份、扫描或改变远端。实际本地释放后才关闭明确归属远端
服务、下载日志并审计。沿用服务72/80GiB、辅助3/4GiB和全部磁盘保护。

若假设不成立，按观测到的主要部分选择下一假设，不扫描缓存阈值；若成立，
先推导有界复用及真实物理容量/依赖合同，再最小验证及普通完整4000回放。
7B W1/W2/G2、3B及基线、dLoRA、消融/机制/敏感性、成本与最终交付仍未完成。

## 依据与接口

沿用 D518_NATIVE_STEP_BOUNDARIES.md 的版本源码核查；本轮再次读取
[vLLM V2 runner](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/model_runner.py)
与[PyTorch原生HOST计数](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/cuda/memory.py)。
配置 configs/ieee_tc/20261014_d519_7b_full_w0_prefix1.yaml；既有 runner 的 d519
分支。原始结果 results/ieee_tc/p2_backend_qualification/d519_20261014；
资格清单 paper_results/ieee_tc/p2_backend/d519_step_components。
使用 run-experiment/vLLM 技能保留标准入口和测量边界，github-sync 明确路径
备份；旧环境例子/通用SLO不覆盖已冻结协议，不 stage 用户 manifest。
