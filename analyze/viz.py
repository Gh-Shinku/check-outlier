import os

import matplotlib.pyplot as plt
import numpy as np

MODULE_MAP = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
    "lm_head": "lm_head",
}
MODULE_ORDER = ["q", "k", "v", "o", "gate", "up", "down", "lm_head"]


def _parse_layer(name):
    for suffix, short in MODULE_MAP.items():
        if name.endswith(suffix):
            if "layers" in name:
                parts = name.split(".")
                idx = parts.index("layers")
                return int(parts[idx + 1]), short
            elif name == "lm_head":
                return -1, "lm_head"
    return None, None


def plot_architecture_heatmap(all_results, save_dir="plots"):
    keys = {}
    for (d1, d2), results in all_results.items():
        for name, m in results.items():
            li, mt = _parse_layer(name)
            if li is None:
                continue
            key = (li, mt)
            if key not in keys:
                keys[key] = []
            keys[key].append(m["jaccard"])

    layer_ids = sorted(set(k[0] for k in keys if k[0] >= 0))
    if any(k[0] == -1 for k in keys):
        layer_ids.append(-1)
    n_layers, n_mods = len(layer_ids), len(MODULE_ORDER)
    data = np.full((n_layers, n_mods), np.nan)
    for (li, mt), vals in keys.items():
        i = layer_ids.index(li)
        j = MODULE_ORDER.index(mt)
        data[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_mods))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(MODULE_ORDER, fontsize=9)
    ylabels = [str(x) if x >= 0 else "LM" for x in layer_ids]
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("Module type")
    ax.set_ylabel("Layer depth")
    fig.colorbar(im, ax=ax, label="Mean Jaccard (6 domain pairs)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "architecture_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_spearman_by_depth(all_results, save_dir="plots"):
    depth_mods = {}
    for (d1, d2), results in all_results.items():
        for name, m in results.items():
            li, mt = _parse_layer(name)
            if li is None or li < 0 or mt == "lm_head":
                continue
            key = (li, mt)
            if key not in depth_mods:
                depth_mods[key] = []
            depth_mods[key].append(m["spearman"])

    layer_ids = sorted(set(k[0] for k in depth_mods))
    mods = [m for m in MODULE_ORDER if m != "lm_head"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(mods)))
    for mi, mt in enumerate(mods):
        xs, ys_min, ys_max, ys_mean = [], [], [], []
        for li in layer_ids:
            key = (li, mt)
            if key not in depth_mods:
                continue
            vals = depth_mods[key]
            xs.append(li)
            ys_mean.append(np.mean(vals))
            ys_min.append(np.min(vals))
            ys_max.append(np.max(vals))
        ax.plot(xs, ys_mean, label=mt, color=colors[mi], linewidth=1.5)
        ax.fill_between(xs, ys_min, ys_max, alpha=0.15, color=colors[mi])

    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, label="Goal (0.8)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "spearman_by_depth.png"), dpi=150)
    plt.close(fig)


def plot_magnitude_scatter(all_outliers, save_dir="plots"):
    domains = sorted(all_outliers.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(domains)))

    layer_data = {}
    for domain in domains:
        for name, info in all_outliers[domain].items():
            li, _ = _parse_layer(name)
            if li is None or li < 0:
                continue
            if li not in layer_data:
                layer_data[li] = {}
            mag = info["values"].float().mean().item()
            layer_data[li][domain] = mag

    layer_ids = sorted(layer_data.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    for di, domain in enumerate(domains):
        xs, ys = [], []
        for li in layer_ids:
            if domain in layer_data[li]:
                xs.append(li)
                ys.append(layer_data[li][domain])
        ax.scatter(xs, ys, label=domain, color=colors[di], s=15, alpha=0.7)

    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Mean magnitude of top-0.1% channels")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "magnitude_scatter.png"), dpi=150)
    plt.close(fig)
