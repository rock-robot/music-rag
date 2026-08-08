"""plot_intervals.py — publication-style interval histogram, corpus vs A / A' / B.

Numbers are the Week 4 interval-histogram table (signed intervals, clamped +/-12).
Replace the four lists with your own if they've changed. Saves a PNG for the README.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# signed interval bins actually present in the table
bins   = [-12,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,12]
corpus = [0.010,0.005,0.025,0.007,0.053,0.035,0.049,0.180,0.098,0.053,
          0.099,0.189,0.050,0.029,0.047,0.007,0.025,0.006,0.006,0.005,0.010]
A      = [0.006,0.001,0.008,0.002,0.045,0.032,0.066,0.228,0.092,0.034,
          0.082,0.226,0.072,0.033,0.037,0.003,0.010,0.004,0.005,0.003,0.008]
Aprime = [0.006,0.001,0.011,0.002,0.036,0.040,0.066,0.224,0.084,0.020,
          0.082,0.248,0.069,0.032,0.039,0.006,0.013,0.006,0.001,0.003,0.005]
B      = [0.007,0.001,0.013,0.004,0.030,0.048,0.063,0.203,0.124,0.005,
          0.104,0.210,0.070,0.030,0.044,0.009,0.015,0.002,0.005,0.002,0.002]

# palette: corpus as a calm reference band, the three systems as distinct lines
INK, GRID = "#1b2130", "#d7dce4"
C_CORP, C_A, C_AP, C_B = "#b8933f", "#4c72b0", "#55a868", "#c44e52"

x = np.arange(len(bins))
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=140)

# corpus drawn as a filled reference so the systems read as deviations from it
ax.fill_between(x, corpus, color=C_CORP, alpha=.18, zorder=1, label="Corpus (reference)")
ax.plot(x, corpus, color=C_CORP, lw=1.6, zorder=2)

for y, c, lab in ((A, C_A, "A  (baseline)"),
                  (Aprime, C_AP, "A\u2032 (length control)"),
                  (B, C_B, "B  (retrieval)")):
    ax.plot(x, y, color=c, lw=1.9, marker="o", ms=3.2, zorder=3, label=lab)

ax.set_xticks(x)
ax.set_xticklabels([f"{b:+d}" if abs(b) != 12 else f"{b:+d}\u2009±" for b in bins], fontsize=8)
ax.set_xlabel("Melodic interval (semitones, clamped \u00b112)", fontsize=10)
ax.set_ylabel("Probability", fontsize=10)
ax.set_title("Interval distribution: generated systems vs. the composer's corpus",
             fontsize=12, color=INK, pad=12)

ax.set_ylim(0, 0.30)                        # headroom so nothing crowds the top
ax.grid(axis="y", color=GRID, lw=.8, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK, labelcolor=INK)
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=1)

# annotate the two bins that carry the story, kept clear of the legend and lines
ax.annotate("B nearly eliminates\nrepeated notes",
            xy=(9, 0.006), xytext=(7.4, 0.155), fontsize=8, ha="center", color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=.9))
ax.annotate("all systems over-produce\nstep-wise motion (\u00b11, \u00b12)",
            xy=(11, 0.212), xytext=(15.5, 0.275), fontsize=8, ha="center", color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=.9))

fig.tight_layout()
fig.savefig("interval_histogram.png", bbox_inches="tight", facecolor="white")
print("saved interval_histogram.png")