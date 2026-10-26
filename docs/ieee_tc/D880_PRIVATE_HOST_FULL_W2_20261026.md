# D880：匹配初始化后的私有CPU构造完整W2

本键只在D879完整实测、严格导出且资源释放后启动。当前是条件性登记，不
以文件存在声称已经启动或完成；以raw/start_receipt、launch和completion为准。

同一D873/D874候选、D875原生资格，继承D876全部Full设置；只将服务/准备
profile和admission长度绑定更新为D879严格匹配身份，并使用独立输出/cache
目录。长度均值来自原完成长度证据，不使用新请求未来生成长度。三实例上限、
共同4000请求、W2 rotation100、prompt/targets、真实remote交付、内存/CPU、
共同SLO和Resident预算全部不改，不引入新参数搜索或短Full。

## 必要核验

复用D726已有运行时解析/严格loader检查：Full runtime和dedicated child的
模型身份必须等于D879 contract；服务及准备profile的SHA和context必须完全
匹配。此为修复D876曾漏掉的绑定检查，不重新测SLO/Resident或整套资格。
配置恢复旧profile和输出路径后必须与D876逐字段一致。D878 collector修复
只影响初始化测量，不作为生产优化收益归因。

## 问题与判定

检验把CPU私有对象构造从core同步事务分离，是否改善完整churn负载的联合
SLO与尾TTFT，及是否增加GPU生命周期成本。不能由65.8ms局部重叠或单次
准备速度推断完整效果。最近完整参照D870为3773/4000=94.325%，
U=11931.565094108693 GPU-s；共同W2预算12083.063870235463 GPU-s。
仍至少3800条联合达标，不改阈值、不补同键重复直到获胜。

结果保留原生token/adapter引用、CPU构造started/settled/failed/discarded、
真实获取次数和字节、完整allocation/release、取消与资源事件；原计算脚本
复用。独立数值正确性未证明时N_correct保持未知，不将绑定成功写作全部
数值证明，也不由一条W2点宣称双模型跨场景G1/G2全部完成。无CI、暂不画图。

raw：`results/ieee_tc/p2_backend_qualification/d880_20261026`；curated：
`paper_results/ieee_tc/p2_backend/d880_private_host_full_w2`。
旧147投稿项、用户manifest和D876–D879失败/测量全部保留。启动前明确文件
备份可回退，最终V2同步仍待完成。两模型同版本矩阵、其余baseline/dLoRA、
A1–A5/S1–S13、两轮审稿图文、数值证据和最终交付继续开放。
