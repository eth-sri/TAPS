import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sb


rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Palatio']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

colors = ["#56641a", "#c0affb", "#00678a", "#984464", "#5eccab", "#cdcdcd"]
c1 = colors[4]
c2 = colors[1]
c3 = colors[2]
c4 = colors[3]
c5 = colors[5]

fig_height = 3.0

def plot_approx_error(baseline_model, ax):
    root = f"exact_tightness/{baseline_model}_results"
    root = os.path.join(os.path.dirname(__file__),root)

    IBP_bounds = (np.load(os.path.join(root, "IBP.npy"))).reshape(-1, 9)
    PGD_bounds = (np.load(os.path.join(root, "PGD.npy"))).reshape(-1, 9)
    TAPS_bounds = (np.load(os.path.join(root, "TAPS.npy"))).reshape(-1, 9)
    MILP_bounds = (np.load(os.path.join(root, "MILP.npy"))).reshape(-1, 9)
    SABR_bounds = (np.load(os.path.join(root, "SABR.npy"))).reshape(-1, 9)
    print(f"Robust Accu:", (MILP_bounds.max(1)<0).mean())

    # The estimated largest logit gap
    PGD_diff = PGD_bounds.max(1)-MILP_bounds.max(1)
    filter = PGD_diff <= 0 # numerical error
    PGD_diff = (PGD_bounds.max(1)-MILP_bounds.max(1))[filter]
    IBP_diff = (IBP_bounds.max(1)-MILP_bounds.max(1))[filter]
    TAPS_diff = (TAPS_bounds.max(1)-MILP_bounds.max(1))[filter]
    SABR_diff = (SABR_bounds.max(1)-MILP_bounds.max(1))[filter]

    ax.vlines(0, 0, 100, color="black", lw=1, zorder=11)
    ax.hist(IBP_diff, bins=bins, density=True, color=c1, histtype="step", label="IBP", lw=3)
    ax.hist(PGD_diff, bins=bins, density=True, color=c2, histtype="step", label="PGD", lw=3)
    ax.hist(SABR_diff, bins=bins, density=True, color=c4, histtype="step", label="SABR", lw=3)
    ax.hist(TAPS_diff, bins=bins, density=True, color=c3, histtype="stepfilled", label="TAPS", lw=3, zorder=10)

mode = "TAPS" # IBP or TAPS
if mode == "IBP":
    fig, ax = plt.subplots(figsize=(5/4*fig_height, fig_height))
else:
    fig, ax = plt.subplots(figsize=(7/4*fig_height, fig_height))

fontsize=18

ax.set_facecolor( (0.98, 0.98, 0.98) )
ax.set_yticks([], [])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.label.set_size(20)
ax.yaxis.set_label_coords(0.0, 1.03)
plt.setp(ax.get_xticklabels(), fontsize=fontsize)
plt.setp(ax.get_yticklabels(), fontsize=fontsize)
if mode == "IBP":
    bins = np.arange(-2.01,2.001,0.02)
else:
    bins = np.arange(-2.025,2.001,0.02)

plot_approx_error(mode, ax)

ax.set_ylabel("Frequency", fontsize=fontsize, rotation=0, ha="left")
ax.set_xlabel("Bound Tightness")
if mode == "IBP":
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([-0.5,0,0.5], [-0.5,0,0.5], fontsize=fontsize)
    ax.set_ylim(0, 50)
else:
    ax.set_xlim(-1, 2)
    ax.set_xticks([-1,0,1,2], [-1,0,1,2], fontsize=fontsize)
    ax.set_ylim(0, 20)

if mode == "SABR":
    ax.legend(bbox_to_anchor=(1.05, 0.9), fontsize=fontsize, frameon=False)

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__),"..","figures",f"exact_tightness_{mode}.pdf"))


# ------- plot effect of TAPS over IBP + STAPS over SABR ------- #
def plot_diff(models, ax, method="TAPS"):
    for i, m in enumerate(models):
        root = f"exact_tightness/{m}_results"
        root = os.path.join(os.path.dirname(__file__),root)
        IBP_bounds = (np.load(os.path.join(root, "IBP.npy"))).reshape(-1, 9)
        TAPS_bounds = (np.load(os.path.join(root, "TAPS.npy"))).reshape(-1, 9)
        SABR_bounds = (np.load(os.path.join(root, "SABR.npy"))).reshape(-1, 9)
        STAPS_bounds = (np.load(os.path.join(root, "STAPS.npy"))).reshape(-1, 9)

        TAPS_effect = (TAPS_bounds - IBP_bounds).reshape(-1)
        STAPS_effect = (STAPS_bounds - SABR_bounds).reshape(-1)

        effect = eval(f"{method}_effect")
        # ax.vlines(0, 0, 100, color="black", lw=1, zorder=11)
        if method == "TAPS":
            bins = np.arange(-10,1,0.1)
        else:
            bins = np.arange(-10,1,0.05)
        ax.hist(effect, density=True, color=eval(f"c{i+1}"), histtype="step", label=m, lw=3, bins=bins)
        # sb.histplot(effect, ax=ax, color=eval(f"c{i+1}"), label=m, stat="density", kde=True, fill=False)

methods = ["TAPS","STAPS"]
for method in methods:
    if method != "STAPS":
        fig, ax = plt.subplots(figsize=(5/4*fig_height, fig_height))
    else:
        fig, ax = plt.subplots(figsize=(7/4*fig_height, fig_height))

    fontsize=18

    ax.set_facecolor( (0.98, 0.98, 0.98) )
    ax.set_yticks([], [])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.label.set_size(20)
    ax.yaxis.set_label_coords(0.0, 1.03)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)

    plot_diff(["IBP", "SABR", "TAPS", "STAPS"], ax, method=method)
    if method == "STAPS":
        ax.legend(bbox_to_anchor=(1.05, 0.9), fontsize=fontsize, frameon=False)
        ax.set_xlabel("STAPS $-$ SABR")
        ax.set_xlim([-1.5,0.1])
        ax.set_xticks(np.arange(-1.5,0.1,0.5))
    else:
        ax.set_xlabel(f"TAPS $-$ IBP")
        ax.set_xlim([-4,0.3])
        ax.set_xticks(np.arange(-4,0.1,1.))
    ax.set_ylabel("Frequency", fontsize=fontsize, rotation=0, ha="left")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),"..","figures",f"{method}_effect.pdf"))
