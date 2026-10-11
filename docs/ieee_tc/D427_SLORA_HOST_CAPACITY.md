# D427：S-LoRA 全池 HOST 容量检查

2026-10-11。CPU 离线审计完成；不是新的 GPU 性能实验。
沿 D425 历史复用、D426 原生计量进入共同部署接入前，发现必须处理的容量问题。
PrimeLoRA D371/D397、ServerlessLLM D414、vLLM D423 均不重跑。

## 已确认的事实

复用原 7B 全部 500 个逻辑 adapter 和冻结内容索引，读取当前 config 与
safetensors headers；128,000 个 FP16 Q/K/V/O A/B 张量，350 个 rank 8、
150 个 rank 16。单份张量合计 **10,905,190,400 B = 10.15625 GiB**。
全物化目录的 12,985,984,450 B 还包括 tokenizer/config 等，不能作为 HOST
权重大小。相同/全零权重仍按每个逻辑 adapter 计，不借内容去重缩减负载。

当前原生 TP1、非 GQA、swap 路径先保存八个独立的分模块 pinned HOST
tensor，再创建独立的 `w_combined_home`，两者均保留。
这一结构已存在于[官方固定版本加载器](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/models/peft/layer_weights/lora_layer_weight.py)；
历史本机 TP/GQA 兼容修改没有删除该分支中的副本。
加载完原 checkpoint 的局部 `weights` 被删除，不等于 layer 中的这些持久字段被释放。

| 完整池副本数 | 仅两份 HOST adapter 张量的必要下界 | 相对 80 GiB 上限 |
|---:|---:|---|
| 1 | 20.3125 GiB | 其他内存尚未计入，不代表合格 |
| 2 | 40.625 GiB | 同上 |
| 3 | 60.9375 GiB | 同上 |
| 4 | **81.25 GiB** | 仅此项已经超限，不启动 |

下界不包含 Python/router/tokenizer、backbone staging、allocator 缓存和取整、
临时 concat、page cache 等。不能靠顺序启动、压缩远端传输或 2 GiB swap
消除四副本同时保留的 pinned 权重。该结论针对当前具体表示和完整池部署，
不是 S-LoRA 理论容量、调度效果或所有配置不可行的结论。

## 验证与证据边界

- 复用 D425 列出的 107 份日志，逐份 SHA 一致。RSS/VmRSS/RssAnon/RssFile、
  `memory.current/peak/max`、MemoryMax、resident-set 模式没有匹配项。
  这是指定文件/模式的结果，不声称全服务器不存在其他测量。
- 既有 API `print_mem_stats` 输出来自 ModelProphet 估算，不是实测 HOST RSS；
  历史 ready/成功记录不能证明共同 80 GiB 限制下可运行。
- 原生 Python 3.9.25 中 21 项测试通过（1.553s）；全套 148 项 CPU 回归通过
  （8.081s）。新增 6 项包括源布局身份、容量边界、缺张量/错误几何拒绝和
  **直接执行未修改原生 loader 的小张量存储检查**：九块独立 storage，合计
  等于两份权重。测试将 `pin_memory` 替为 CPU clone，CUDA 未初始化；因此
  证明表示/引用关系，不冒充实际 pinned RSS、GPU 或数值推理资格。
- 当前 helper 只新增纯容量下界计算和三个布局 SHA 常量；不是已接通的服务
  启动器，不授权任何部署。所有返回值保持 `deployment_memory_qualified=false`。
- header/config 审计复用此前内容 SHA，不重读 10 GiB payload，不宣称本轮重新
  校验了完整权重。没有新工件池、trace、远程发布或下载、GPU 作业、图或清理。

原始证据：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d427_20261011/`。
`audit.json` SHA：`6aa7f573cb48a72e66417d87e3b4c5c9792ed00a10fd3944d24f3a7c8bc5f977`。
容量 recipe 复用 D425 inventory、冻结内容索引和现有 source-adapter helper。
当前嵌套源码及已有十文件兼容补丁不修改。

官方固定 commit 的 layer loader 和 HF loader 经 HTTPS 获取，与 Git HEAD
逐字一致，`sources.json` 保留两者与当前兼容源码的独立 SHA。完整证据压缩包
`paper_results/ieee_tc/baseline_audit/d427_slora_host_capacity.tar.gz` 共 12 个普通
文件、21,008 B，逐成员与 raw 字节核对一致，SHA：
`2eaa2248bbdb4bc2dfb13043f389b272c9ac4bbd396dbdc862cf24e602368946`。
147 个保护项零变化；四个本轮 CPU scope 均核对身份、全部进程退出后关闭，
high/max/OOM/swap 事件为零。测试与 header 审计有短暂并行，两个 scope 的
各自峰值为 213,282,816 / 59,113,472 B（其和也低于 4 GiB）；没有重型任务。

## 下一步及主线约束

优先核查 packed 原生执行实际使用的 HOST 表示，验证能否用不改变权重、
paging、调度、加载时机的表示适配消除冗余副本；若不成立，使用公开允许的
较少副本配置并检查其剩余内存预算，不能尝试已证明超限的四副本原表示。
不做跨逻辑 ID 内容去重、不缩减 500 池、不增加内存额度或免费预加载。
此处尚未选择/实现适配，也未冻结 1/2/3 副本为性能最优。

随后回到既有共同服务/pre-ready/physical-owner 集成，显式 S-LoRA validator，
受影响 100 请求、缺失 500-ID/1,000 资格和一次完整 W0；复用已通过局部证据。
保持 60 秒共同通知、固定业务时刻、全池 HOST 准备计量和真实资源释放。
后续 7B 基线、W1/W2/G2、3B、全部批准消融/敏感性和文档仍 OPEN。
