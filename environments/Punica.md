# Punica Environment

Punica is kept as a scoped Llama-2 7B auxiliary baseline only.

## Current Status

- Upstream repo: `/home/qhq/serverless_llm_baselines/repos/Punica`
- Project entry: `/home/qhq/serverless_llm_baselines/Punica_project`
- Current use: limited Llama-2 7B replay / sanity reference.
- Not used as a full four-backbone main-table baseline.

## Boundary

Punica should not be expanded by changing its core serving algorithm. Any future
use must stay in wrapper/materialization/replay/summary layers and must clearly
state the limited coverage.
