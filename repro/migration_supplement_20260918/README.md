# Supplemental recovery backup (2026-09-18)

Read RECOVERY_GUIDE.md before restoring. This supplement extends the previous
repro/migration_20260918 snapshot without changing the source project's files.

file_backup/files.jsonl, when present, maps original paths to checked gzip objects.
Use restore_file_backup.py first without --restore-to to verify all objects.
For baseline raw evidence and independent projects, use file_backups/ in the
private QHQsky5295/server-migration-backup-20260918 repository's main branch.

Environment source overlays can be checked with apply_source_overlays.py.
unrecorded_sources records additional startup hooks and local import paths;
review and remap machine-specific absolute paths before installing them.

Model weights, frozen training products, omitted large assets, inaccessible
files and credentials still require separate backup. No fresh-host GPU
reproduction was performed. Original project files and environments were not
modified, and no experiments were started or stopped by this backup.
