# 服务器迁移恢复指南（2026-09-18 补充）

本次补充保存可读取的项目恢复材料、原始实验记录、模型配套配置、源代码版本、系统配置参考和环境源码修改。模型权重、冻结训练产物、大型二进制资产、受权限保护的目录和凭据仍需另行备份。只有文件清单的资产不代表已保存内容。

## 1. 获取正确的代码版本

先在新机器配置 GitHub 访问权限。私钥和访问令牌不包含在这些仓库中。

从私有总仓库 `QHQsky5295/server-migration-backup-20260918` 的 `main` 分支取得本指南、工具和 `recovery-projects.json`。各项目的新增内容位于迁移分支，默认分支可能没有这些材料。

```bash
python3 restore_projects.py --map recovery-projects.json --destination /new/server-projects
# 检查计划后，显式创建新的项目目录：
python3 restore_projects.py --map recovery-projects.json --destination /new/server-projects --apply
```

该工具只克隆并定位指定提交，不安装依赖、不启动服务、不覆盖已有目录。总仓库本身用本次推送完成后报告中的提交定位。

## 2. 校验并恢复补充文件

每个项目的材料放在 `repro/migration_supplement_20260918/`。文件内容按 SHA256 去重，并使用标准 gzip 压缩；多个压缩对象顺序存放在 `packs/part-*.bin` 中。`files.jsonl` 记录每个原文件的相对路径、哈希、权限、时间、压缩对象和偏移。部分较早采集的对象单独位于 `objects/`。

可在仓库根目录执行 `sha256sum -c repro/migration_supplement_20260918/SHA256SUMS`，核验补充材料本身。

先校验备份。默认命令只读取，不修改任何项目：

```bash
python3 restore_file_backup.py --backup /new/project/repro/migration_supplement_20260918/file_backup
```

再将文件恢复到该项目的克隆目录：

```bash
python3 restore_file_backup.py \
  --backup /new/project/repro/migration_supplement_20260918/file_backup \
  --restore-to /new/project
```

工具会先校验全部对象。目标文件已经相同则跳过；只有目标内容匹配清单记录的原 Git 版本时才允许更新不同内容，否则拒绝覆盖。归档记录中的路径不能越出目标目录，目标路径不能穿过软链接。文件所有者、ACL 和扩展属性未由此工具恢复。

基线实验的补充数据以及无独立可写仓库的小项目存放在私有总仓库的 `repro/migration_supplement_20260918/file_backups/项目名/`，用同一工具恢复到对应项目目录。这样，尚未公开的实验数据保存在私有仓库中。

`symlinks.json` 是路径关系记录，包含未备份资产相关的链接；工具不会自动创建它们。先迁移对应资产，再按新服务器目录重新建立必要的链接。FaaSLoRA 当前工作区的 `data`、`artifacts`、`results` 原来指向历史工作区；必须保留这种对应关系。

## 3. 重建实验环境

上一次提交的 `repro/migration_20260918/environments/` 保存包清单和安装来源。本次 `environments/` 增加了关键实验依赖的 RECORD 哈希核对、激活脚本和补充环境资料。

1. 依据系统参考配置安装兼容的操作系统、GPU 驱动、CUDA 和编译工具。`system/` 中的挂载 UUID、用户名、路径及服务设置需要适配新机器，不要直接覆盖 `/etc`。
2. 在新前缀中使用相应 `conda-explicit.txt` 重建 Conda 包，再审查安装 `pip-install.txt`。完整的 `pip-versions.txt` 也包含 Conda 管理的包，不宜全部重复覆盖安装。
3. 本地可编辑依赖按 `upstream-sources.json` 和补充 `source_repositories/revisions.json` 中的版本恢复；应用已经保存的源码补丁、未跟踪源码和补充文件。
4. 按项目上次说明恢复直接修改过的安装包源码。`apply_source_overlays.py` 默认只检查；只有原始安装内容与记录的哈希匹配才允许 `--apply`。
5. 挑战杯项目 `.deps/qwen35_training` 含混合版本元数据，优先使用已保存的精确源码归档；其他环境记录不等于离线环境镜像。

本次检查覆盖若干关键实验依赖的文本源码，未逐字节验证全部安装文件。历史环境中发现的缺失文件按原样记录，不能据此认定所有历史环境可直接运行。依赖下载地址在未来是否仍可访问尚无保证。

## 4. 单独迁移仍缺少的资产

参考本次排除清单与上次 `asset-transfer-required.json`。模型权重、冻结 LoRA、检查点、大型原始数据和容器卷均不能以目录清单代替。凭据应另行安全保管。

可重复生成的运行缓存没有纳入备份。旧资产目录中存在独特的配置和训练产物，不能仅因年代较早就删除。文件采集期间若发生改动或读取失败，会记入对应 `capture-exclusions.json`。

## 5. 验收后再退役旧机器

- 校验所有新归档、备份对象和独立迁移资产。
- 执行项目的环境导入检查和小规模真实实验，验证数据、权重和路径确实可用。
- 对需继续训练的项目，实际加载检查点，并验证训练状态能够恢复。
- 对论文实验，核对结果、图表、参数和所需原始证据；保留已有校验差异的记录。

本次对备份工具进行了恢复、损坏检测和防误覆盖检查，未在一台清空的新服务器上重建全部环境或重跑 GPU 实验。完成上述验收后，才能确认旧服务器可退役。
