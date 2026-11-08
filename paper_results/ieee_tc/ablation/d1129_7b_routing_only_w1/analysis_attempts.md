# D1129离线分析记录

全部分析在4000请求终态、GPU实际释放、本机服务域移除与远端inactive后
执行。15:49:13核对tmux pane114467已结束且退出0；launch的service、replay、
watchdog退出值均0。无第二次性能回放，无运行中生产/远端修改。

各CPU单元前缀`primelora-tc-d1129-`，后缀`-20261108.service`：

| 单元中段 | 工作 | 主进程 | wrapper | 结果 |
|---|---|---:|---:|---|
| metrics | D1113原生与物理计量 | 0 | 1 | 完成保存；既有inotify ENOSPC导致管理收尾timeout |
| review | D299合同/资源/远端 | 0 | 1 | 完成保存；同类管理收尾timeout |
| stages | D455分阶段分析 | 0 | 1 | 完成保存；同类管理收尾timeout |
| compare | D1119/D1120方法与既有Full锚点 | 0 | 0 | 4000输入、4000实际路由通过，208输出digest差异保留 |
| postrun-sources | 来源/旧稿/用户dirty保护 | 0 | 1 | 84来源、144旧稿文件、3用户文件通过；管理收尾timeout |

所有分析CPU2/3/26/27，MemoryHigh3GiB/Max4GiB、swap0、CUDA隐藏。
确认前一主进程结束且实际cgroup无PID后才开始下一分析；没有因为wrapper
timeout重复已成功完成的分析或更改全局inotify设置。最后15:56:22核对
postrun-sources MainPID0、ControlGroup空，Invocation
d0326f13c1d8494088a251ff2be7b76b，实际域已移除。

metrics/review/stages的主进程Invocation分别为
e272549221824ebf97b6150ec3dfdda8、3265bc4561b34ec08edd82d4b37aa8c5、
278358f6899f4c7ca63e4ba4dacd7e44；compare为90469bbd192d4cc78362f0985ccdb7dc。
compare.py在回放前只准备，实际释放后才AST检查与执行；未更改已封存脚本。

远端132条journal先核对本轮clock与SHA，再复制并建既有D299要求的parent
链接；原始路径和SHA记入RAW/remote_journal_source.json。未重复打包、未
重新发布工件、未从时延扣除真实传输。无新增负载或权重。

无独立语义评审，执行者检查不冒称独立评审。保留n_correct未知、输出差异
及资源/质量代价，单轮不计算CI。不按胜负选择保存结果。
