# D527：新测量绑定后的 Full 原始100请求资格

2026-10-14。当前为准备与 CPU 资格完成，尚未启动 GPU 回放。

## 问题与唯一变化

沿 D520–D526 的已记录假设：原生 LoRA 元数据中已在 CPU 上已知的映射，
不必先交给 GPU 计算再读回作 CPU 分支。保留原生 GPU buffer、CUDA Graph、
slot 生命周期和失效语义；固定双缓冲实际 pinned 增量99040 B计入 HOST 预算。
官方源码、局部算术和原生调用证据见 D520/D522/D525/D526 文档，不重复运行。

D523 失败是旧准备时间 profile 与新后端身份不匹配，发生在请求执行之前；
不构成性能退化。D527 复用 D526 成功标定的200类服务、24类准备时间。
除两个 profile 绑定和独立输出/cache路径外，新配置与 D523 完全相同。
与 D476 的机制候选差异仍仅为 cpu_live_mapping_v1 及其必要匹配的时间初始化。
不改九公式、公共 SLO、GPU/HOST 预算、输入、生成协议或控制参数。

## 已完成证据

- 308项 CPU/configuration/启动入口检查通过，无 failure/error/skip，无 CUDA 初始化。
- 按实际子进程模型身份加载两个 D526 profile 成功，非目标配置相等。
- CPU 实际资源域清空并关闭；峰值3083395072 B，high/max/OOM/kill/swap均0。
- 旧投稿147文件和用户 manifest不变；本次不生成新权重、trace或工件池。
- 小型证据：paper_results/ieee_tc/p2_backend/d527_metadata_full_qualification/。

## 下一次运行及验收

既有 run_d293_7b_prefetch_validation_100.sh 的 d527 入口，原 W0 前100请求，
完整 tc_ieee_full 路径、真实远端只读发布对象、相同60秒准备窗口和资源包络。
只运行一次；启动前备份及新鲜资源检查，结束后核验真实释放再停止本次远端。
检查100条请求的原生数量/身份/计时、metadata实际安装/使用/预算计量，
并区分 native binding/count 与独立数值正确性证明（n_correct仍未知）。
这不是最终 G1/G2 性能点，不依据短负载胜负筛选。完整合同通过后才进入
一个新的4000请求 W0；沿用冻结 V1+单轮V2，与保留的 D476及发表系统比较。

共享 k3s 属于其他项目，完全不修改。HydraServe D510隔离容器的4000请求
结果继续复用；全部pp1不能宣称证明 pipeline consolidation。
7B W1/W2/G2、3B、dLoRA B6、消融和敏感性、审稿文档及图表仍未闭口。
