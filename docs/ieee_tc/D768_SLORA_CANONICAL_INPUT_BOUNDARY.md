# D768：S-LoRA 3B公共输入序列的必要兼容适配

2026-10-23。只修复D767确认的输入差异，不优化baseline性能，不改变SLO。

## 依据与范围

D767前100请求中99完成原生合同，req_00028失败。CPU使用原公共准备器与
原生tokenizer复现，只有这一条出现89/91输入token差异，首差异索引34。
原生Transformers4.34.0/tokenizers0.14.1。失败运行与完整诊断原样保留。
历史GQA/BMM路径及D762共同tokenizer资产核查不曾消除此原生重分词差异；
本次新证据不是性能不好，而是执行工作量不一致。

固定官方源码的HTTP manager在encode之后将ID列表交给router：
[S-LoRA manager](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/httpserver/manager.py)。
[官方API](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/api_server.py)
的流式入口接收文本。本次把共同准备器已生成的ID跨过这一旧分词入口，
保留router之后原生代码。没有升级专用后端或改请求、adapter、到达、目标长度。

## 显式接口

- 默认`native_text_v1`完全保留旧路径；D768显式采用`canonical_token_ids_v1`。
- `/generate_stream`传入完整token列表、文本SHA和ID列表SHA；不允许缺字段后
  回落到旧tokenizer。此专用view不声称支持未经适配的其他文本生成端点。
- 原生HTTP manager验证输入契约、整数ID与词表范围，再执行原生input/context/
  token容量检查，交给同一个router队列。复制列表以避免调用者后续修改。
- 实际原生forward仍记录它真正收到的ID、adapter、进程和输出token；未将发送
  的预期数量冒充实际输出，也未放宽旧观察器对输入的逐项核验。
- 原生调度、KV、LoRA、BMM、采样、输出和资源生命周期不变。3B BMM仍不是
  7B packed BGMV统一分页路径，不作完整机制覆盖或独立数值正确性声明。
- source view与配置绑定input_mode；原始checkout不写入修改，历史view不覆盖。

## 验证与后续

CPU相关57测试通过：生成源码AST还原后与原路径一致；实际API→manager
传递精确ID且不重分词；原生三个容量限制保持；错误hash/ID拒绝；传输与外层
服务接线、实际响应核验保持。首轮测试中两个新fixture错误已修正，失败日志保留。
CPU域peak349380608B，high/max/OOM/swap均0并已关闭；尚未启动D768 GPU。

继续复用D767相同100请求、29实际adapter、17369目标tokens、500静态池，
仅以新键检验受影响GPU输入边界。随后补尚缺的500-ID覆盖、1000兼容及完整
W0/W1/W2；不重复未受影响基线，不因结果差继续搜baseline配置。新观测决定
是否通过；D767失败不是S-LoRA本身的性能缺陷证据。
