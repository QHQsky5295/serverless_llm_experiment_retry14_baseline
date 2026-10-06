# D303：同配置完整 W0 验证确认状态复用

## 启动前登记

2026-10-07，继续 7B 主线。复用 D302 已完成的 100 请求机制验证和 D301
371 项回归测试，不重复短测、warm、准备 profile 或数值诊断。
本轮为完整 4,000 请求的开发验证，不是留出测试或正式 G1/G2 验收。

唯一实现变化是 D301 的同身份 confirmed scope 发布修正：同 owner/clock/
epoch/source identity 的记录不再被发布动作立即清除；身份变化仍失效，
选中来源的保护与复核不变。不增加 TTL、并发、扩缩容阈值或其他参数。
历史与官方源码依据、可证伪假设见 D301_CONFIRMED_SCOPE_PUBLICATION.md。

对照为既有 D300 完整 W0。配置逐字段检查：仅输出、NVMe 冷缓存和 HOST
冷缓存路径不同；模型、生成、输入、profile、全部控制参数一致。
修正的源文件 SHA：
`d6ec59a9877023eafc9686251bdebce48a470e3af6c55efcc76fe2df8e3d2078`。
新配置 `configs/ieee_tc/20261007_d303_7b_full_w0.yaml` SHA：
`18a5d633fac9a5cf13d4be260d323bb7796ad8698c4d7fd2e0493de141fcb3e5`。

## 合同与待判断问题

- 原 source42 的完整 4,000 请求、500 逻辑 adapter，W0 open-loop，60 秒通知。
- 同 fixed_length_greedy_v1、真实远端 prepublished_gzip_v1、已批准只读缓存。
  不制作新工件、请求内打包或复制整池；真实传输照常计量。
- 同 D300 cap8/slots8/rank16、.92 显存比例、TP1/FP16 和 D298 profiles。
- 同 72/80 GiB 服务内存、2 GiB 服务 swap、4 GiB 外置工具上限；150/100 GiB
  磁盘保护不降低。一次仅一个重型实验，无同步压缩/构建/安装。
- 比较物理 GPU-s、平均/P95 TTFT、TPOT、严格等待分解和路由观测证据。
  检查是否真的复用，以及控制代价是否抵消收益；全部退化保留。
- 最终视图含原生读取的请求比例不是 RPC 总量，条件组差异不是因果收益。
- 只给本次完整运行的观测，不对单次结果计算 CI 或宣布显著性。
- D170 仅提供临时 timing-only 上界；最终数值正确性、共同 warm/Resident
  仍开放，N_correct 不从 token 数或绑定检查推断。

## 来源、保护与收尾

计划 SHA `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`；
冻结 V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
完整计划、V1、执行记录均已重新核对。保护清单仍为 146/147，唯一既有
baseline 脚本差异及 SHA 未变；原始 pass=false 保留，不称全部一致。
用户生成 manifest 不修改、不 stage；历史结果不覆盖。

原始结果专属目录：`results/ieee_tc/p2_backend_qualification/d303_20261007/`。
使用既有 runner 参数 d303，tmux `primelora-d303-7b`；禁止覆盖旧执行键。
结束后先确认物理 GPU 释放，再按捕获的 InvocationID 停止远端服务；保存
传输与监控日志并核对 SHA。复用现有完整回放 analyzer，另建 curated 输出。
本轮不画图；7B 未验收前不进入 3B、外部 baseline 或后续消融敏感性。

当前状态：启动前检查点 `26f0b3fa133c53e9ad2bedb9857d6844e71f2a39` 已推送
并核对远端；06:53:45+08 启动完整回放。实际句柄及进度在 EXECUTION_STATUS.md。
运行未结束，尚无完整性能结论；不得重复启动同一执行键。
