"""Plot D219 descriptive evidence; explicitly not a cross-system comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
curated = json.loads((HERE / "curated/d219_full_curated.json").read_text())
out = ROOT / "figs/ieee_tc/p2_backend/20261005_d219_full"
out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.4, "axes.titlesize": 8.2, "axes.labelsize": 7.4,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
})

blue = "#2F5597"
gold = "#C28E0E"
gray = "#A6A6A6"
tiers = curated["dispatch_tier_counts_request_level"]
names = ["GPU", "HOST", "NVMe", "Remote"]
counts = [tiers.get(name.lower(), 0) for name in names]
metrics = curated["descriptive_metrics"]

fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.72), gridspec_kw={"wspace": 0.60})
axes[0].bar(names, counts, color=[blue, "#5B9BD5", "#70AD47", gold], width=0.65, edgecolor="none")
axes[0].set_title("(a) Dispatch tier\n(n=4,000)")
axes[0].set_ylabel("Requests")
axes[0].set_ylim(0, max(counts) * 1.18)
axes[0].tick_params(axis="x", rotation=22)
axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.45)
axes[0].set_axisbelow(True)

latency_names = ["TTFT", "service\nTTFT", "E2E"]
latency = [metrics["p95_ttft_ms"], metrics["p95_service_ttft_ms"], metrics["p95_e2e_ms"]]
axes[1].bar(latency_names, latency, color=[blue, gray, gold], width=0.65, edgecolor="none")
axes[1].set_title("(b) P95 latency\n(descriptive)")
axes[1].set_ylabel("Milliseconds")
axes[1].set_ylim(0, max(latency) * 1.18)
axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.45)
axes[1].set_axisbelow(True)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
fig.savefig(out / "fig_d219_confirmed_snapshot_full_diagnostic.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "fig_d219_confirmed_snapshot_full_diagnostic.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
(out / "fig_d219_confirmed_snapshot_full_diagnostic_manifest.json").write_text(json.dumps({
    "kind": "diagnostic_figure_v1",
    "source": "paper_results/ieee_tc/p2_backend/20261005_d219_full/curated/d219_full_curated.json",
    "scope": "complete single-seed exploratory replay; no cross-system or held-out multi-seed claim",
    "caption": "D219 provides descriptive dispatch-tier and latency evidence after the confirmed-snapshot cache path completed 4,000 requests. Bars are not a formal comparison.",
}, indent=2) + "\n")
