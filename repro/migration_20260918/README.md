# Server migration snapshot — 2026-09-18

Original workspace: `/home/qhq/serverless_llm_baselines`

Original commit: `d85263e00e976cc61c8938e662d77a2b249cdf58`

Backup branch: `migration/server-20260918`

This branch preserves the selected working files plus environment records. The
running server's source files, active branch, original Git index, installed
packages and closed experiment results were not rewritten by this backup.

## Restore order

1. Clone this repository and check out the backup branch. Use
   `working-file-inventory.json` to verify the captured working files.
2. Restore each required environment into a NEW prefix. For Conda environments,
   `conda create --prefix NEW_PREFIX --file environments/NAME/conda-explicit.txt`
   records the exact Conda packages. Review `pip-install.txt` and install those
   packages with that environment's Python, using the appropriate CUDA wheel
   index when needed. `pip-versions.txt` is the complete inventory, including
   Conda-owned packages; it is not a blanket installation command.
3. Recreate local/editable dependencies from `upstream-sources.json` (where
   present) and `manual-source-installs.json`. Check out each recorded commit,
   apply its binary-capable patch, and restore listed untracked source files.
   Local origins in the source inventory refer to another captured checkout.
4. Apply installed-package changes with `apply_source_overlays.py --snapshot
   environments/NAME --prefix NEW_PREFIX`. The default only checks. Add
   `--apply` after validation; it refuses unexpected package versions/content.
5. Recreate data/model/result locations and links from `external-links.json`.
   Absolute paths are recorded for identification and must be remapped if the
   new server uses different directories. A symlink is not an asset backup.
6. If `config_snapshots/lora_manifest_1000.local.json` exists, it preserves the
   server-local generated manifest without replacing the canonical experiment
   configuration. Select it explicitly for the matching historical workload.
7. Run project smoke checks, then a small real workload before full experiments.

## Material outside Git

Model weights, frozen trained adapters, large original datasets, credentials,
and most raw runtime result directories remain on the original server unless
explicitly represented by a captured file/archive. They require a separate
asset transfer. Do not erase the source server after only cloning this branch.
Rebuildable caches were excluded from this code snapshot; nothing was deleted.

`excluded-working-files.json` lists selection exclusions and permission errors.
Existing tracked files are inherited from the parent commit. The inventory
does not claim access to other users' protected directories or container data.

## Validation boundaries

See `validation.json`. Original result-bundle checksums may already be stale;
their failures are recorded as found rather than rewriting scientific evidence.
Snapshot-specific `SHA256SUMS` protects the newly captured migration material.
The GPU experiments have not been rerun, and dependency downloads on a new
server have not been tested.
