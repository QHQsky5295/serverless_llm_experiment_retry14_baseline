# D635：以同一次已持有副本确认构造 HOST 替换来源

2026-10-17。D634 完整 W0 中止的正确性修复；不是性能改进结论，未启动 GPU。
九公式、冻结 V1／单轮 V2、两副本起点和全部模型参数不变。

## 失败日志给出的直接证据

流式读取 D634 已封存 main_outcome，核对原 SHA 后检查362个准备计划、387次
尝试的3552条文件 fallback。发现唯一一条不匹配，恰属唯一的
native_outcome_unresolved 尝试；其余仅表示这些字段一致，不代替数值正确性。

| 项目 | 先前快照 | 持有引用的确认 |
|---|---:|---:|
| security_lora_0060，NVMe 分配文件字节 | 37,748,736 | 37,752,832 |
| 差值 | — | +4,096 B |
| 内容、路径、adapter、owner、tier、表示 | 一致 | 一致 |
| 引用状态 | — | held，content_verified=true |

plan_id 为2ebe256f915147408bf6d40e035ecdcd；attempt_id 为
03e3e6f0bcb646c480a0ee50eb874c7d。保留完整 source/reference 对和原始来源 SHA，
不重新扫描现场来替代失败时观察，不把本次未知的内核转换原因说成已证实。

D634仍为797 arrived、796 submitted、774原生合同完成、总4000请求未完成。
Uobs=2016.6970647417475 GPU-s，完整U和N_correct未知，不参加达标排名。
全部服务和远端已关闭；没有内存超限/OOM、外部GPU作业或请求期打包证据。

## 历史与原理

1. 774bd7f引入HOST替换的完整文件fallback，并严格校验它与持有引用一致。
2. D310／4ea7daa已区分内容身份与分配观察：acquire_confirmed在owner锁内可
   更新已收费范围内的物理分配量；来源身份、签名和容量校验仍严格。
3. D365对需求读取的确认沿用了这一原则；D606/D610减少观察遍历，没有授权
   跨观察拼接字段。D628候选已由D631撤销，不重新引入其私有验证改动。
4. 本次调用方先取source_snapshot，再acquire_confirmed，却仍用前者构造
   fallback的物理量、后者提供引用证明。两次观察合法不同就会触发严格拒绝。

根因定位为调用方混合观察，而不是已经证明文件被损坏或引用失效。修复在
acquire_confirmed成功后，用它返回的confirmed_source构造完整fallback；
owner也来自同一held receipt。不会再次读取一个更晚快照，也不放宽下游断言。
原子持有、未知RPC结果保留引用、明确回复后释放、替换收益公式均不改变。

## 本轮原始资料核查

- [Linux FIEMAP](https://docs.kernel.org/filesystems/fiemap.html)：调用之间文件
  映射可变化；SYNC会同步文件。本次不新增fsync、sleep或轮询，不据此推断
  D634究竟发生了哪一种内核映射变化。
- [vLLM 0.30.0 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  LRU添加路径先验证加载再驱逐，原生pin接口独立于普通缓存查找；参考其
  “确认拥有的对象再提交替换”的边界，而不是照搬LRU或冒称它管理本项目文件。
- [vLLM 0.30.0 model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)：
  CPU/GPU对象管理与文件来源是不同层的责任。本次只修正文件receipt衔接。

## 最小验证及失败不隐藏

两次最初的测试夹具均未复现目标异常：一次尝试让8字节文件的分配量降至
预留数据下界以下，一次发现其分配上界没有额外4KiB。这是夹具条件不适用，
不是生产修复失败；两份日志及身份、关闭记录保留，不放宽生产容量检查。

随后把问题分开验证：既有ConfirmedFilePublication测试负责真实微型文件的
保守分配边界及合法刷新；新增组合测试只控制调用者较早收到的分配观察，
实际获取引用仍通过真实文件owner。修改生产代码之前，它复现完全相同的
HOST eviction fallback lacks its held exact file copy异常。它不是内核事件重放。

新增负向检查逐一破坏footprint、held状态、verified、adapter、owner、path、
tier、content和representation，要求原严格检查全部拒绝；不增加容差或兜底。
同时复用实际组合路径的取消、失联回复、旧引用保护及释放检查。

## 下一步与结论边界

CPU回归结束后记录真实结果并提交/推送。通过只允许进入新执行键的同配置
完整W0，不允许把774请求当完整结果或直接拼接D632/D633旧成功点。之后核验
跨场景影响与独立数值正确性，按协议确定可复用范围。没有新调参、阈值变化、
性能结论或作图。全部7B/3B、dLoRA、A1–A5、S1–S13、成本和审稿文档交付保留。

原始诊断：results/ieee_tc/p2_backend_qualification/d635_20261017/；
失败来源继续在D634目录，零覆盖。

## CPU 资格结果

180项transfer-pressure、144项GPU引用、215项request-lifecycle检查分别在
独立进程完成，合计539项通过。新增组合测试及9种负向字段破坏全部通过；
既有真实文件合法分配刷新、越界拒绝、内容/epoch失效、取消与失联回复的
所有权检查继续通过。测试夹具中native对象受控，不冒称原生GPU数值验证。

回归资源域峰值574631936B，high/max/OOM/kill/swap均0；按实际Invocation、
inode、递归空进程核验后关闭，路径消失。所有五个辅助域均已关闭，没有GPU
或远端服务运行。生产仅改变调用方fallback记录构造；residency_manager与
preloading_planner保持原SHA。接下来备份，完成资源/输入资格后才启动新W0。
