"""Make a compact, explicitly diagnostic D217 evidence figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
curated = json.loads((Path(__file__).parent / "curated/d217_retry10_curated.json").read_text())
out = ROOT / "figs/ieee_tc/p2_backend/20261005_d217_retry10"
out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

blue = "#2F5597"
gray = "#A6A6A6"
gold = "#C28E0E"

fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.15), gridspec_kw={"hspace": 0.62})

# Historical values are shown only as descriptive diagnostics; they are not a
# paired comparison and are intentionally labelled as such in the manifest.
labels = ["D215", "D216", "D217"]
means = np.array([236.004, 238.515, 87.591])
p95 = np.array([947.056, 690.333, 371.939])
x = np.arange(len(labels))
w = 0.30
axes[0].bar(x - w / 2, means, w, label="mean", color=[gray, gray, blue], edgecolor="none")
axes[0].bar(x + w / 2, p95, w, label="Type-1 P95", color=["#D9D9D9", "#D9D9D9", gold], edgecolor="none")
axes[0].set_xticks(x, labels)
axes[0].set_ylabel("Control path (ms)")
axes[0].set_title("(a) Control-path diagnostic")
axes[0].set_xlabel("Diagnostic run (100 requests)", labelpad=1)
axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.45)
axes[0].set_axisbelow(True)
axes[0].legend(frameon=False, loc="upper right", ncol=1, handlelength=1.1, borderpad=0.15)
axes[0].set_ylim(0, 1030)

inv = curated["population_and_invariants"]
names = ["success", "contract", "snapshot", "owner/epoch", "tier"]
values = [
    inv["successful"], inv["output_contract_match"], inv["confirmed_dispatch_snapshot"],
    inv["selected_intent_receipt_owner_match"], inv["source_tier_equals_dispatch_tier"],
]
axes[1].barh(np.arange(len(names)), np.array(values, dtype=float), color=blue, height=0.62, edgecolor="none")
axes[1].set_yticks(np.arange(len(names)), names)
axes[1].invert_yaxis()
axes[1].set_ylabel("Invariant")
axes[1].set_title("(b) Request-level invariants")
axes[1].set_xlabel("Requests passing", labelpad=1)
axes[1].set_xlim(0, 108)
axes[1].set_xticks([0, 50, 100])
axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.45)
axes[1].set_axisbelow(True)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, pad=1.5)

fig.savefig(out / "fig_d217_confirmed_snapshot_diagnostic.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "fig_d217_confirmed_snapshot_diagnostic.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
(out / "fig_d217_confirmed_snapshot_diagnostic_manifest.json").write_text(json.dumps({
    "kind": "diagnostic_figure_v1",
    "source": "paper_results/ieee_tc/p2_backend/20261005_d217_retry10/curated/d217_retry10_curated.json",
    "scope": "100-request mechanism diagnostic; not a formal cross-run comparison",
    "historical_bars": {"D215": "prior single diagnostic", "D216": "prior single diagnostic"},
    "caption": "D217 confirms request-level owner/epoch and dispatch-tier invariants. Historical control-path bars are descriptive only and are not a paired causal comparison.",
}, indent=2) + "\n")
