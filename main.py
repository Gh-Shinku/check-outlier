import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import torch

from data.loader import DOMAIN_CONFIG, load_domain
from data.preprocess import tokenize_texts
from infer.model import load_model, load_tokenizer
from infer.hook import ActivationStatsHook
from analyze.stats import compute_outliers
from analyze.viz import plot_heatmap, plot_overlap_matrix
from verify.compare import compare_domains

DOMAINS = sorted(DOMAIN_CONFIG.keys())
PERSIST_KEYS = ["model", "max_samples", "batch_size", "seq_len", "top_k_pct"]


def _layer_sort_key(layer_name):
    parts = layer_name.split(".")
    for p in parts:
        if p.isdigit():
            return (int(p), layer_name)
    return (-1, layer_name)


class RunManager:
    def __init__(self, base_dir="runs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.run_dir = None

    def create_run(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, ts)
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"Run directory: {self.run_dir}")
        return self.run_dir

    def domain_dir(self, domain):
        d = os.path.join(self.run_dir, domain)
        os.makedirs(d, exist_ok=True)
        return d

    def stats_path(self, domain):
        return os.path.join(self.domain_dir(domain), "stats.pt")

    def plots_dir(self):
        d = os.path.join(self.run_dir, "plots")
        os.makedirs(d, exist_ok=True)
        return d

    def save_stats(self, domain, stats):
        path = self.stats_path(domain)
        torch.save(stats, path)

    def load_stats(self, domain):
        path = os.path.join(self.run_dir, domain, "stats.pt")
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        return None

    def has_stats(self, domain):
        return os.path.isfile(os.path.join(self.run_dir, domain, "stats.pt"))

    def completed_domains(self):
        return [d for d in sorted(os.listdir(self.run_dir))
                if os.path.isdir(os.path.join(self.run_dir, d))
                and os.path.isfile(os.path.join(self.run_dir, d, "stats.pt"))]

    def save_args(self, args_dict):
        path = os.path.join(self.run_dir, "args.json")
        with open(path, "w") as f:
            json.dump(args_dict, f, indent=2)

    def load_args(self):
        path = os.path.join(self.run_dir, "args.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def list_runs(base_dir="runs"):
        if not os.path.isdir(base_dir):
            return []
        entries = []
        for name in sorted(os.listdir(base_dir), reverse=True):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                domains = sorted(
                    d for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d)) and d != "plots"
                )
                entries.append((name, domains))
        return entries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Activation Outlier Persistence Experiment"
    )
    parser.add_argument(
        "--resume", nargs="?", const="", default=None,
        help="Resume latest run. Optionally specify a run timestamp."
    )
    parser.add_argument(
        "--model", default="/home/zyt/sda_ws/models/LLaMA-3.1-8B-Instruct",
        help="Model path or name"
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=128,
        help="Samples per domain (default: 128)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Inference batch size (default: 1)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048,
        help="Sequence length for tokenization (default: 2048)"
    )
    parser.add_argument(
        "--top-k-pct", type=float, default=0.001,
        help="Top-k percentile for outlier definition (default: 0.001)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runner = RunManager()

    # ── resolve run directory ──
    if args.resume is not None:
        runs = runner.list_runs()
        if not runs:
            print("No previous runs found. Starting fresh.")
            runner.create_run()
        else:
            if args.resume:
                run_ts = args.resume
            else:
                run_ts = runs[0][0]
            runner.run_dir = os.path.join(runner.base_dir, run_ts)
            if not os.path.isdir(runner.run_dir):
                print(f"Run '{run_ts}' not found. Starting fresh.")
                runner.create_run()
            else:
                completed = runner.completed_domains()
                saved_args = runner.load_args()
                for k, v in saved_args.items():
                    setattr(args, k, v)
                print(f"Resuming run: {run_ts}  (cached: {completed}, args: {saved_args})")
    else:
        runner.create_run()
        runner.save_args({k: getattr(args, k) for k in PERSIST_KEYS})

    # ── collect or run stats per domain ──
    domains_to_run = [d for d in DOMAINS if not runner.has_stats(d)]
    cached_domains = [d for d in DOMAINS if runner.has_stats(d)]

    for d in cached_domains:
        print(f"  {d}: cached, skipping inference")

    if domains_to_run:
        print(f"Loading model {args.model} ...")
        model = load_model(args.model)
        tokenizer = load_tokenizer(args.model)
        hook = ActivationStatsHook(model)

        for domain in domains_to_run:
            print(f"\n{'=' * 60}")
            print(f"Processing domain: {domain}")
            print(f"{'=' * 60}")

            hook.reset()
            texts = load_domain(domain, max_samples=args.max_samples)
            print(f"  Loaded {len(texts)} samples")

            input_ids = tokenize_texts(tokenizer, texts, seq_len=args.seq_len)
            print(f"  Tokenized to shape {list(input_ids.shape)}")

            device = next(model.parameters()).device
            n_batches = (len(input_ids) + args.batch_size - 1) // args.batch_size
            with torch.no_grad():
                for i in range(0, len(input_ids), args.batch_size):
                    batch = input_ids[i: i + args.batch_size].to(device)
                    model(input_ids=batch)
                    if ((i // args.batch_size) + 1) % 10 == 0:
                        print(f"  Batch {i // args.batch_size + 1}/{n_batches}")

            stats = hook.get_stats()
            runner.save_stats(domain, stats)
            print(f"  Saved stats ({sum(1 for s in stats.values() if s['count'] > 0)} layers active)")

        hook.remove()
    else:
        print("All domains already cached. Skipping inference.")

    # ── load all stats and compute outliers ──
    all_outliers = {}
    for d in DOMAINS:
        stats = runner.load_stats(d)
        if stats:
            outliers = compute_outliers(stats, top_k_pct=args.top_k_pct)
            all_outliers[d] = outliers
            print(f"  Loaded {d}: {len(outliers)} layers with outliers")

    if len(all_outliers) < 2:
        print("Need at least 2 domains with stats for comparison. Exiting.")
        return

    # ── cross-domain comparison ──
    all_results = {}
    domain_list = sorted(all_outliers.keys())
    for i in range(len(domain_list)):
        for j in range(i + 1, len(domain_list)):
            d1, d2 = domain_list[i], domain_list[j]
            results = compare_domains(all_outliers[d1], all_outliers[d2])
            all_results[(d1, d2)] = results

    # ── export CSV ──
    csv_path = os.path.join(runner.run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain_1", "domain_2", "layer", "overlap", "jaccard", "spearman"])
        for (d1, d2), results in sorted(all_results.items()):
            for layer, m in sorted(results.items(), key=lambda x: _layer_sort_key(x[0])):
                w.writerow([d1, d2, layer, f"{m['overlap']:.3f}", f"{m['jaccard']:.3f}", f"{m['spearman']:.3f}"])
    print(f"\nResults saved to {csv_path}")

    # ── plotting ──
    sample_layers = list(next(iter(all_outliers.values())).keys())[:10]
    plots_dir = runner.plots_dir()
    plot_heatmap(all_outliers, sample_layers, save_dir=plots_dir)
    plot_overlap_matrix(all_outliers, sample_layers, save_dir=plots_dir)
    print(f"Plots saved to {plots_dir}/")

    # ── summary ──
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    overlaps, jaccards, spearmans = [], [], []
    for pair, results in all_results.items():
        for _, m in results.items():
            overlaps.append(m["overlap"])
            jaccards.append(m["jaccard"])
            spearmans.append(m["spearman"])

    if overlaps:
        lines = [
            f"Mean overlap:  {np.mean(overlaps):.3f}  (goal >0.7)",
            f"Mean Jaccard:  {np.mean(jaccards):.3f}  (goal >0.6)",
            f"Mean Spearman: {np.mean(spearmans):.3f}  (goal >0.8)",
            f"Total layer-pairs compared: {len(overlaps)}",
        ]
        for line in lines:
            print(f"  {line}")

        summary_path = os.path.join(runner.run_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("Summary\n")
            f.write("=======\n\n")
            f.write("\n".join(lines))
            f.write("\n")
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
