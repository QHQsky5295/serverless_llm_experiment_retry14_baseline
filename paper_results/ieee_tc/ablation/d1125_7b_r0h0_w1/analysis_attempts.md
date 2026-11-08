# D1125 离线分析尝试记录

性能回放仅一次，4000请求完成、真实GPU释放且三项运行退出码均0。
下述调用/管理问题不属于性能重跑，不删除原测量、不更换比较点。

- `run-u385.service`：复用D1113收集器，main/wrapper均0，5.155秒。
- `run-u386.service`：D299入口未设置项目PYTHONPATH，在import阶段退出1；
  未生成review。仅修正分析调用环境，不改分析器或生产代码。
- `run-u387.service`：D299已进入检查，但既有接口要求campaign父目录中的
  `remote_transfers.jsonl`；当时只在run目录保存原始journal，故退出1。
  未生成review。按D1124已有布局新增指向本轮journal的软链接，未复制或改写
  测量字节。其资源域已空；wrapper最后退出1，runtime105.270秒。
- `run-u388.service`：上述路径修正后，D299全部检查通过并保存review；
  main退出0。但systemd出现control/memory inotify ENOSPC及stop-sigterm
  timeout，wrapper退出1，runtime114.305秒。实际资源域先已为空，之后移除；
  这是既有管理层监听额度问题，不能改写为wrapper全通过，也不等同内存OOM
  或测量失败。未更改全局限制、共享服务或再次重复该检查。
- `run-u389.service`：D455阶段分解通过，main/wrapper均0，27.390秒。
- `run-u390.service`：当前7B两机制/四格分析通过，main/wrapper均0，84.741秒。
- `run-u391.service`：64个唯一来源SHA、144个旧投稿保护文件及用户dirty
  manifest校验通过，main退出0；实际域先为空后移除。wrapper最终timeout/
  退出1，runtime110.391秒；与u388同类管理收尾问题，保留此限制，不重复
  来源检查、不混为独立评审或测量OOM。

全部分析使用CPU2,3,26,27、MemoryHigh3GiB/Max4GiB、swap0且隐藏GPU，
均在性能服务实际退出后执行。未把4000请求当作4000独立重复。
