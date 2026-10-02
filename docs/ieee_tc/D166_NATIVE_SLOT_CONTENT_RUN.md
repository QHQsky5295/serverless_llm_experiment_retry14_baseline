# D166：7B 原生槽位内容实际核验

## 问题与预先固定的范围

使用 D165 已测试并备份的独立检查入口，不修改服务策略或再次回放 Full。
问题是：在对应 adapter 保持引用、原生 scheduler 已排空时，实际 GPU 槽位的
A/B 张量及 padding、缺失 slice 是否与已注册 CPU 张量精确一致。
生成前后分别检查；不用输出相似性、概率容差或逻辑名称替代内容证据。

固定原 seed42 trace 的前 22 个请求，不重建 trace 或权重。这是包含首个
`medical_lora` 的最短原始前缀，同时含 `finance_lora`、rank 8/16 零权重。
覆盖 12 个逻辑 adapter、4 个权重 SHA，不称完整 500-ID 功能验收。
该选择依据静态工件属性，未查看本次运行结果。配置复用 D156 的 TP=1、
max_num_seqs=4、max_loras=4、FP16；诊断串行执行，不测并发吞吐。

使用本地既有冻结工件，是明确的内容诊断，不是远端性能运行或远端失败后的
回退；D78/D80 的一次性交付缓存已完成、保持不动，远端服务本次不启动。
诊断中的同步屏障和 GPU→CPU 读回开销不进入普通 Full 路径或性能排名。

## 证据边界

| 项目 | 当前状态 |
|---|---|
| 检查器 CPU 资格 | 复用已封存 D165；不重复执行 |
| 输入、配置和 147 项历史保护 | D166 检查通过，原始输入 SHA 一致 |
| slot_content1 | 模型加载前失败：环境 HOST 策略未写入模型配置；无 GPU 样本 |
| slot_content2 | 完成实际加载，但首个内容检查因未覆盖原生 unpacked 张量布局而拒绝；无生成成功样本 |
| 检查器布局补全 | 定向 17 项通过（0.044 s）；回归 815 项通过（74.700 s），两组有重叠 |
| slot_content3 | 完成：22/22 请求、44 次快照、19,888 个槽张量比较，不一致元素 0 |
| checkpoint→已注册张量 | 尚未闭合，不由本项反推 |
| 每 token 的实际执行映射/算术 | 尚未闭合，前后快照不代表生成期间 |
| 完整正确性、common warm-SLO、G1/G2 | 尚未闭合 |

依据计划第十一节和 academic-plotting，本项交付正确性状态表，不画性能排名。

## 资源与来源

父代码检查点 `17468f24b517eb255eb1df67b9999dcda3608834`；本次只补全独立
诊断的两处函数及测试，不修改普通 Full 服务策略或 IEEE 行间公式。
原始证据目录 `results/ieee_tc/p2_backend_qualification/d166_20261002/`。
输入回执 SHA `5cc582ee4ff99715a4da6a5544a1cae5cc5deb357dd32a517aa1a56f420d5007`。
继续采用推理服务 72/80 GiB、swap 2 GiB；外置工具 3/4 GiB、swap 0，
真实 worker 隔离、独立 watchdog、实际持卡释放以及 150/100 GiB 磁盘门槛。

启动前复用 D152 公共下载缓存审核，精确删除 5 份可再获取归档及 5 份响应头，
释放 2,192,306,176 分配字节。公共注册表 SHA/大小/可获取性、文件身份、打开
引用、硬链接和项目引用检查通过；安装环境、编译缓存、模型和结果未删除。
本次独立允许清单 SHA `1a8e073911745e4c99528c080eaef1ad34e93ee4220b3dcb553d256ef8f64557`。
实际 audit/apply invocation 为 `afa6b35235c64ce09994457bf49a8726` /
`5df38101aa634183a8666a9c142f70de`，均退出 0 且资源域已移除。

首次运行的失败为 `native HOST allocator policy must be in the frozen model
configuration`，发生于子运行时创建前。D156 bootstrap 原本从 profiling spec
注入四项 HOST 配置；slot-content 模式没有该 spec，原始 parent YAML 本身
不含它们。新配置显式复用这四项，并核对 D164 Full 的相同取值，不改检查、
不改分配策略或内存额度。失败原始结果 SHA
`b3fa1a0721d888690ee20f7147a9e1d3dab3351f3c05b0c6701c9d9499370ab0` 保留。
失败服务 invocation `67f7a71e34004794aa3f93d9d1427061`；退出 2，watchdog
退出 0，实际资源域移除及 GPU 空闲已核验，不作为系统性能失败或成功样本。

第二次实际 inventory 含 452 个槽张量，包括 `lm_head` 的两个 raw tensor、
embedding 的 3-D A/4-D B；D165 只遍历线性层的 tuple/list，因此在首个
`slot_content_before:req_00000` 拒绝，并未测出内容不一致。
实际持卡 48.535512155 秒、一个 GPU，已确认释放；不计为服务性能结果。
原始 SHA `0cd93e13150957ec4ad0dc5ae456f01f3aa41f3b3cc4b7be91ca7c704fa7ce38`。

检查器现在显式覆盖已由 HOST contract 验证官方 resetter 的 **absent unpacked**
模块：对整个矩阵检查零值，3-D embedding A 以 `target[slot]` 读回；其他
4-D buffer 仍为 `target[slot,0]`。有权重的 unpacked setter 仍拒绝，不推断
未经验证的转置/分片行为；不跳过 embedding/head，不修改原生 setter 或服务
路由，默认 Full 不调用诊断。新增反例覆盖这四个 raw buffer 的非零残留、
populated unpacked 拒绝及 CPU tensor 不能冒充 GPU 证据。

依据实际 0.30.0 源码与在线官方
[embedding 实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/layers/vocab_parallel_embedding.py)。
官方 logits 网页抓取失败，改用已安装固定版本文件核查，不声称联网读取成功。
安装文件 SHA：embedding `fce09cb442fe2d8409c367b99e0885b8dfda9fbe1ec708a8dd4b90d4b96e1d84`；
logits `dc7ae097ad47628061a53aec2c8b2d603dd8b38fe20c955b7d48e9248de029d1`。

## 实际结果与结论

| 观测项 | 结果 | 含义/限制 |
|---|---:|---|
| 原始请求 / 成功数 | 22 / 22 | 原生生成数量等于目标，逐请求原始身份一致 |
| 逻辑 adapter / 权重 SHA | 12 / 4 | 含 finance、medical 非零工件和 rank 8/16 零工件 |
| 生成前后快照 | 44 | scheduler 排空、引用持有、显式设备同步 |
| 槽张量比较 | 19,888 | 每快照 452 个，覆盖实际槽位 0、1、2、3 |
| 逐元素不一致 | 0 | 包括 padding 及缺失 slice；所有值有限 |
| absent-module 张量比较 | 8,624 | 包括 176 次 embedding/head raw buffer 检查，不跳过未应用的模块 |
| 单 GPU 生命周期 | 387.762329310 s | 包含诊断读回与同步，不能作为正常性能样本 |
| 资源采样 / 服务内存峰值 | 399 / 5,702,578,176 B | 最低主机可用内存 106,507,923,456 B |
| high/max/OOM / swap | 均为 0 | 一张 GPU、服务域与外置域均已释放 |

支持的结论仅为：这组现有工件在本次实际执行中，生成前后已注册 CPU 张量
与对应 GPU 槽内容精确一致，且未发现缺失模块或 padding 非零残留。
它不认证 checkpoint→native registry 的所有转换、不证明每个 token 期间的
执行映射与算术，也不能用 44 次重复快照充当独立性能重复。
完整正确请求数仍为未知，不据此重标 D164 的 4,000 请求为数值合格。

前两次失败均保留：首次在模型创建前拒绝配置遗漏；第二次实际持卡后由检查器
拒绝未覆盖布局。第三次完成并不删除失败，也不将其错误归因为服务性能。
实际 service invocation 为 `404399e467314cc6bb45913a2ec127e3`；服务、
watchdog 均退出 0。资源释放与两个资源域消失于 18:08:14 +08 核验。
结果验证在独立 3/4 GiB、swap=0 域完成，耗时 12.41 s，最大 RSS 1,265,692 KiB。

CSV/JSON：`paper_results/ieee_tc/p2_backend/20261002_d166_slot_content.*`。
下一步仍按 Prime7B 主线补必要的 checkpoint/执行映射、common warm-SLO 和
新指标下旧/新 Prime 对照；不重复本项已完成诊断，不宣称 G1/G2 达标，不先跑
3B 或外部基线。D78/D80 一次性交付缓存和旧投稿结果始终未动。
