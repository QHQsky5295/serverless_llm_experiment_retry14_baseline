# D528：修复 Full 的测量来源绑定，不改变服务策略

2026-10-14。D527 已失败并释放资源，失败记录永久保留。

## 原因与检查缺口

D527 在 ScenarioRunner 初始化加载服务 profile 时拒绝配置，尚未提交请求。
最终 YAML 的 ieee_admission_profile 仍带旧 model_backend_id/profile_id；
D524 为 metadata 策略生成的两个标识已经改变，并被 D526 原生标定实际使用。
两个配置的 profile_means、window_s、transfer_limit 完全相同。
来源见 prepare_d298_capacity.py 的 derive_source_runtime 分支及
ieee_tc_preflight.py 原生完成长度的 model_backend_id/profile_id 构造；
不是测量值不匹配，也不能删除严格身份检查来绕过。

D527 的 CPU 检查在测试内用 development_contract 替换了 admission profile，
因此只证明替换后的配置可加载，没有证明最终 YAML 可加载。这是本次检查的
真实缺口；308项通过不能继续被解释成实际完整入口已经通过。

## 最小修复和验证

新 D528 YAML 只将两个来源标识绑定到实际 D526 标定身份，另使用新输出/cache
目录。所有数值、公式、预算、SLO、输入、生成、profile 内容和服务代码不变。
不修改 D527 配置或失败原始文件；不重新标定、不新增权重/trace/工件池。

新增四个测试直接从最终 YAML 走正常 scenario 合并和 pending-runtime 解析，
调用 ScenarioRunner 实际两个 loader；没有从测量合同替换测试输入。
覆盖 D527 拒绝复现、D528 双 profile 绑定、唯一差异恰为两个标识，以及
单独篡改任一标识仍拒绝。连同受影响配置/前缀入口共14项通过，无 CUDA 初始化。
已有 D527 CPU smoke、D522设备算术与 D526原生安装/标定证据继续复用，不重复。

D527 本地 service/aux 实际路径消失、GPU释放；远端三服务inactive且MainPID0，
18080/18081无监听。本次启动前链路1000Mb/s/full，失败前无请求传输。
失败及清理关键记录包含在 D528 qualification_evidence.tar.gz。

## 下一步与边界

备份后用既有 runner 的 d528 入口运行一次原100 Full，验证请求合同与实际
metadata安装、预算计量和使用；短负载不作最终 G1/G2 排名或胜负筛选。
合格后一个新的4000 W0，按冻结 V1+单轮V2判断收益，保留所有失败和退化。
独立全模型数值正确性仍未知，不由token数量或局部算术结果推断。

共享 k3s 属于其他项目，不操作。HydraServe D510隔离容器结果复用；其全部pp1
的机制范围如实保留。完整7B W1/W2/G2、3B、消融/机制/敏感性和交付仍 ACTIVE。
