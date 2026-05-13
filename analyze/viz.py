import os

import matplotlib.pyplot as plt
import torch


def plot_heatmap(all_outliers, layer_names, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    domains = list(all_outliers.keys())

    for layer in layer_names:
        fig, axes = plt.subplots(
            len(domains), 1, figsize=(14, 2 * len(domains)), squeeze=False
        )
        for idx, domain in enumerate(domains):
            if layer not in all_outliers[domain]:
                axes[idx][0].text(0.5, 0.5, "N/A", ha="center", va="center")
                continue
            data = all_outliers[domain][layer]["max_abs"].cpu().float().numpy()
            axes[idx][0].plot(data, linewidth=0.5)
            axes[idx][0].set_ylabel(domain, fontsize=8)
            axes[idx][0].set_ylim(0, data.max() * 1.1)
        axes[0][0].set_title(layer, fontsize=10)
        fig.tight_layout()
        safe_name = layer.replace(".", "_")
        fig.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=150)
        plt.close(fig)


def plot_overlap_matrix(all_outliers, layer_names, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    domains = list(all_outliers.keys())
    n = len(domains)

    for layer in layer_names:
        if not all(layer in out for out in all_outliers.values()):
            continue
        indices_list = [all_outliers[d][layer]["indices"].cpu() for d in domains]
        matrix = torch.zeros(n, n)
        for i in range(n):
            set_i = set(indices_list[i].tolist())
            for j in range(n):
                set_j = set(indices_list[j].tolist())
                intersection = set_i & set_j
                union = set_i | set_j
                matrix[i][j] = len(intersection) / len(union) if union else 0.0

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix.numpy(), vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(domains, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(domains, fontsize=8)
        ax.set_title(f"Jaccard - {layer}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        safe_name = layer.replace(".", "_")
        fig.savefig(os.path.join(save_dir, f"jaccard_{safe_name}.png"), dpi=150)
        plt.close(fig)
