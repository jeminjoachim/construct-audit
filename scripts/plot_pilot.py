"""
plot_pilot.py — one figure: per seed, the original score, its five padding scores,
and the edited variant(s), seeds ordered by original score.

Reads only results/pilot_score_table.csv (no post text), writes
results/fig_pilot_scores.png and .pdf.

    python scripts\\plot_pilot.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

TABLE = "results/pilot_score_table.csv"
OUT = "results/fig_pilot_scores"

# Frozen PACC reference points (pooled, decision log 2026-09-12)
MEAN_P_NEG, MEAN_P_POS = 0.3025, 0.5202

# Colors: validated categorical pair (blue, orange); padding is a gray context mark
C_ORIG, C_EDIT = "#2a78d6", "#eb6834"
C_PAD, C_RULE = "#8a8984", "#c3c2b7"
INK, INK_2 = "#0b0b0b", "#52514e"
SHORT = {"question_form": "QF", "object_swap": "OS", "object_swap_v2": "OS2"}

t = pd.read_csv(TABLE, dtype={"seed_id": str})
t["include_primary"] = t["include_primary"].astype(str).str.lower() == "true"

orig = t.groupby("seed_id")["score_original"].first().sort_values()
seeds = list(orig.index)                     # lowest original at the bottom
y = {s: i for i, s in enumerate(seeds)}

fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(seeds) + 1.6), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

for ref, lab in [(MEAN_P_NEG, f"typical non-DD post ({MEAN_P_NEG:.2f})"),
                 (MEAN_P_POS, f"typical DD post ({MEAN_P_POS:.2f})")]:
    ax.axvline(ref, color=C_RULE, lw=1, ls=(0, (3, 3)), zorder=0)
    ax.text(ref, len(seeds) - 0.35, lab, ha="center", va="bottom", fontsize=7.5, color=INK_2)

labels = []
for s in seeds:
    rows = t[t["seed_id"] == s]
    yy = y[s]
    pad = rows[rows["variant_id"].str.startswith("padding")]["score"]
    edits = rows[rows["edited_feature"].isin(SHORT) & rows["include_primary"]]
    xs = [orig[s], *pad, *edits["score"]]
    ax.plot([min(xs), max(xs)], [yy, yy], color=C_RULE, lw=1.2, zorder=1)
    ax.scatter(pad, [yy] * len(pad), s=34, facecolors="#fcfcfb", edgecolors=C_PAD,
               linewidths=1.2, zorder=2)
    ax.scatter([orig[s]], [yy], s=70, color=C_ORIG, edgecolors="#fcfcfb", linewidths=1.5, zorder=4)
    for _, e in edits.iterrows():
        ax.scatter([e["score"]], [yy], s=70, marker="D", color=C_EDIT,
                   edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
        ax.annotate(SHORT[e["edited_feature"]], (e["score"], yy), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=6.5, color=INK_2)
    seed_lab = "DD" if str(rows["seed_label"].iloc[0]) == "1" else "not DD"
    labels.append(f"{s}  ({seed_lab})")

ax.set_yticks(range(len(seeds)))
ax.set_yticklabels(labels, fontsize=8, color=INK)
ax.set_xlim(0, 1)
ax.set_ylim(-0.7, len(seeds) + 0.2)
ax.set_xlabel("Classifier score (frozen v1 model)", fontsize=9, color=INK)
ax.tick_params(axis="x", labelsize=8, colors=INK_2)
ax.tick_params(axis="y", length=0)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(C_RULE)
ax.grid(axis="x", color="#ecebe8", lw=0.6)
ax.set_axisbelow(True)

handles = [
    Line2D([], [], marker="o", ls="", color=C_ORIG, markersize=8, label="Original post"),
    Line2D([], [], marker="o", ls="", markerfacecolor="#fcfcfb", markeredgecolor=C_PAD,
           markersize=6.5, label="Original + one neutral sentence (5 versions)"),
    Line2D([], [], marker="D", ls="", color=C_EDIT, markersize=7,
           label="Edited variant (QF question form; OS object swap, exploratory)"),
]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.04), ncol=1,
          frameon=False, fontsize=8, labelcolor=INK, borderaxespad=0)
ax.set_title("Neutral text moves scores more than the codebook's deciding edit",
             loc="left", fontsize=11, color=INK, fontweight="bold", pad=78)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"wrote {OUT}.png and {OUT}.pdf")
