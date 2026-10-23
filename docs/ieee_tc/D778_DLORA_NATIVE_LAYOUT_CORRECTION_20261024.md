# D778：修正测量适配对原生 LoRA 池维度的误读

2026-10-24。D777 检查点推送后，为下一步真实加载核对已有 loader probe 和
原生实现时发现一处具体错误。没有运行 GPU，不存在可用性能结果；不将
适配错误归因于 dLoRA 的机制或性能。

## 发现与修正

D776 新增的 worker receipt 曾检查
`len(cpu_lora_weights) == adapter_count`。实际
[官方 LoRaEngine/LlamaLoRaEngine](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/worker/lora_engine.py)
及保留的本地实现都以模型层为外层列表；每层有 qkv/o/gate_up/down 四组，
每组 A/B 张量的第零维才是 adapter 槽数。3B 的外层为28层，不是500。
错误检查会拒绝合法真实启动。此前合成 fixture 也误用了 adapter-major
形状，故 D776/D777 的 CPU 通过不能证明这个维度正确。

本轮仅修正回执观察：检查层数与原生 `num_hidden_layers` 一致，每层组数
与 `lora_cnt` 一致，各 A/B 张量第零维为完整 adapter 数。回执分别记录
`cpu_lora_layers` 与 `cpu_adapter_capacity`。不改变原生布局、加载器、运算、
调度、放置、迁移、性能参数或既有结果；不加绕过错误的 fallback。

## 针对性验证

新增测试直接提取并执行保留的 LoRaEngine/LlamaLoRaEngine 构造方法，只将
张量存储替换成 shape holder，避免 CPU/GPU 大分配；不是另造一份布局实现。
3B 实际配置为 hidden3072、28层、24 Q heads、8 KV heads、500 adapters、
max_r16。原生构造得到28层，qkv A的形状为(500,3072,48)。

修正后回执接受正确28层/500槽；将任一 A 的第零维改为499即拒绝。
仅重跑受影响的7项 OwnedRuntime 检查，全部通过；D777其余共享入口/生成
检查未改变，不作重复广泛回放。RAW为
`results/ieee_tc/dlora_qualification/d778_20261024/`。

这是原生构造与形状核查，不是数值正确性、实际 GPU 分配或完整服务资格。
使用最新D778 source view及当前helper，不能继续使用引用旧helper SHA的
D776/D777 view启动新实验。旧检查点和其限制原样保留，不追改历史证据。

下一直接进入已接好的原生 DP2/TP1 加载、必要 LoRA/GQA 和共同HTTP资格。
完整3B W1/W2、其余3B baseline、PrimeLoRA优化和全部后续实验主线不变。
