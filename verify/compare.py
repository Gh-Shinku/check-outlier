from scipy.stats import spearmanr


def compare_layer(stats_a, stats_b):
    max_a = stats_a["max_abs"].cpu()
    max_b = stats_b["max_abs"].cpu()

    indices_a = set(stats_a["indices"].tolist())
    indices_b = set(stats_b["indices"].tolist())

    intersection = indices_a & indices_b
    union = indices_a | indices_b
    k = min(len(indices_a), len(indices_b))

    return {
        "overlap": len(intersection) / k if k > 0 else 0.0,
        "jaccard": len(intersection) / len(union) if union else 0.0,
        "overlap_count": len(intersection),
        "spearman": spearmanr(max_a.float().numpy(), max_b.float().numpy()).correlation,
    }


def compare_domains(outliers_a, outliers_b):
    shared = set(outliers_a.keys()) & set(outliers_b.keys())
    return {layer: compare_layer(outliers_a[layer], outliers_b[layer]) for layer in shared}
