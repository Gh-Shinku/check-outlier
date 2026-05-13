import torch


def compute_outliers(stats, top_k_pct=0.001):
    outliers = {}
    for layer_name, s in stats.items():
        max_abs = s["max_abs"]
        if max_abs is None:
            continue
        k = max(1, int(len(max_abs) * top_k_pct))
        topk = max_abs.topk(k)
        mean = s["sum"] / s["count"] if s["count"] > 0 else None
        var = (
            s["sumsq"] / s["count"] - (s["sum"] / s["count"]) ** 2
            if s["count"] > 0
            else None
        )
        outliers[layer_name] = {
            "indices": topk.indices,
            "values": topk.values,
            "max_abs": max_abs,
            "mean": mean,
            "variance": var,
        }
    return outliers
