# D875：私有 CPU LoRA 构造的真实原生资格

2026-10-26。承接 D873/D874，同一候选；九公式、准入策略、工作负载和阈值不变。
本轮复用原生资格入口，仅增加 `native_private_host` 模式，没有新建回放框架。

## 结果与解释边界

原 trace 前四条请求原生输出数为 152/59/123/217，均与目标一致。第二种既有
adapter 在第一条请求首 token 后开始私有 CPU 构造；core 记录 started=1、
settled=1、discarded=failed=in-flight=0。构造完成后确认 HOST-only，尚未占用
GPU slot；随后按需激活。两种 adapter 的注册 CPU 张量与实际 GPU slot 内容
精确检查均通过，4个需求登记、4个GPU引用和1个HOST引用均正常释放。

构造时间与第一条请求首末原生 token 时间区间交集为 **65.801491 ms**。
这是实际区间重叠，不是逐 iteration/kernel 持续推进证明，也不能直接换算为
减少65.8ms延迟。未插入sleep、延长输出、改权重或根据结果重选请求。
完整4000请求的净性能由后续D876决定，不追加短Full回放。

保留 D870 模型配置，仅新增 `private_cpu_core_commit_v1`；实际使用D747
逐张量descriptor、冻结CUDA13工具链、原生allocation、72/80GiB服务包络。
资格只用既有本地冻结工件，不伪作真实远端性能；后续Full仍是真实remote。
有效资格服务内存峰值6004101120B，high/max/OOM/swap0，物理GPU正常释放。
147保护项与用户manifest无变化。独立数值正确性仍未知，3B零权重可区分性
限制保留；这些检查不推导 N_correct，不宣称G1/G2达标。

## 所有尝试保留

1. 继承的旧资格命令使用`--trace`而不是`--replay-trace`，参数解析拒绝，未加载模型。
2. 修正参数后，资格启动器遗漏正式Full已有工具链，原生warmup找不到ninja；
   未进入业务请求，完整释放。不是候选算法性能失败。
3. 复用冻结工具链和descriptor后，新资格入口未登记pending demand而被拒绝；
   保留原生admission配置，没有关闭检查。入口补齐既有登记/退休API。
4. 最终原生资格通过。此前源码快照、测试及所有失败记录均保留，不覆盖。

首次9项/17subtests通过；补齐pending调用后重测受影响3项，通过。此前D874
232项核心所有权/关闭/取消检查复用，不重跑整套资格。此次只改资格入口，
没有改D873/D874生产模块。失败暴露的是本轮入口复用不完整，后续应使用
历史最终有效启动器，不能复用其初始失败版本。

## 依据、证据和下一步

复核官方 [vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
的CPU构造和 [UniProc executor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
同步worker边界；与D871–D874的历史分析一致。未重开整owner线程化、graph
inventory、epoch缓存、阈值修改等已关闭方向。

原始证据：`results/ieee_tc/p2_backend_qualification/d875_20261026/`；汇总及SHA：
`paper_results/ieee_tc/p2_backend/d875_private_host_native/summary.json`、`manifest.json`。
没有新建工件/负载、远端服务或共享k3s变更。

下一步D876：冻结一个完整3B W2新键，只新增该opt-in并更换三个结果/cache路径；
使用原4000请求、W2映射、gzip/isal真实交付、D747、相同预算/初态/后端参数。
以D870的3773/4000、11931.565094 GPU-s作为公开历史开发比较，不改95%及
12083.063870 GPU-s预算，不隐藏退化。所有双模型矩阵、baseline/dLoRA、
A1–A5/S1–S13、独立数值证明、审稿图文和最终V2交付仍开放，目标active。
