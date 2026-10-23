# D783：dLoRA已完成回放的离线汇总适配

本次只扩展`summarize_serverlessllm_replay.py`的显式dLoRA响应身份检查，
不改已执行服务、生成、放置、迁移、配置或原始结果。要求原生sequence
model-ID/index、request-ID、adapter路径/ID、timing contract和正常终态
相符；不能借vLLM名称或HTTP200补过错误绑定。

复用已有静态池summary测试，加入dLoRA正确及错误ID/type/path/terminal
分支。8项summary测试通过；主仓库分析器另外关联原生双worker/GPU租期、
远端500池准备和逐请求计时。两次离线消费者失败记录保留，第三次分析成功；
没有重跑D783 GPU实验。

D783 3B：100/100正常终态，17,369原生token，29个实际served ID；两worker
各加载完整500池，GPU初始放置250/251，CPU KV配置不变。资源818.521554742
GPU-s；共同联合SLO0/100，长准备和前缀积压完整保留。此为短前缀资格，
不是完整W1/W2结果，也不是独立数值正确性证明。详细证据见主仓库
`docs/ieee_tc/D783_DLORA_FULL_POOL_PREFIX100_20261024.md`与D783原始目录。
