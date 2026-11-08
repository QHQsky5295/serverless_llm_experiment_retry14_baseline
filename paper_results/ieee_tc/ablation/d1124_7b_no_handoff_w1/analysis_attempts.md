# D1124 离线分析尝试记录

性能回放仅一次，正常结束。下述失败不属于性能实验失败，不删除、不混入指标。

- `run-u375.service`：复用D1113收集器，main/wrapper均0。
- `run-u376.service`：D299完整请求/计量/传输检查，main/wrapper均0。
- `run-u377.service`：D455阶段分析，main/wrapper均0。
- `run-u378.service`：D1123→D1124比较与机制开关检查，main/wrapper均0。
- `run-u379.service`：D1051物理handoff分析在读取结果前退出1。历史
  `scope_identity`仅接受推理/测试UUID scope，拒绝当前CPU分析的
  `run-u379.service`（`ValueError: not an owned TC service/test UUID scope`）。
  未创建handoff输出、未修改测量或生产代码。新分析脚本改为记录实际
  `cg_path`、unit、PID、UID及分析角色；3/4GiB、swap0、CPU affinity、
  隐藏GPU与无GPU任务断言全部保留，不把分析服务冒称推理scope。
- `run-u380.service`：修正离线运行身份后，物理handoff回执分析通过；
  main/wrapper均0，22.005s，未增加GPU回放。
- `run-u381.service`：68个唯一来源SHA、144个旧投稿保护文件及用户dirty
  manifest校验通过；main/wrapper均0，4.968s。
