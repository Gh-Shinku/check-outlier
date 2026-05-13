# Activation Outlier Persistence Experiment

Verify whether activation outliers in LLMs are **channel-fixed structural properties of weights**, not artifacts of input distribution. Based on the AWQ paper's key observation.

## Hypothesis

If outliers (top 0.1% channels by `max |activation|`) consistently appear in the same channels across code, math, and prose inputs, they are structural properties of the weights. A single pair with overlap <0.3 refutes the claim.

## Quick Start

```bash
make test       # quick: 1 sample per domain
make run        # full: 128 samples per domain
make resume     # resume latest incomplete run
```

## Datasets

| Domain | Source | Field |
|---|---|---|
| wikitext | `Salesforce/wikitext` wikitext-2-raw-v1 | `text` |
| gsm8k | `openai/gsm8k` main | `question` |
| code_alpaca | `flwrlabs/code-alpaca-20k` | instruction+input+output |
| pubmed | `slinusc/PubMedAbstractsSubset` | `abstract` |

## Output

```
runs/<timestamp>/
├── args.json               # run configuration
├── results.csv             # per-layer overlap, jaccard, spearman
├── code_alpaca/stats.pt    # per-channel running stats
├── gsm8k/stats.pt
├── pubmed/stats.pt
├── wikitext/stats.pt
└── plots/                  # heatmaps + jaccard matrices
```

## CLI

```
--model PATH        model path (default: /home/zyt/sda_ws/models/LLaMA-3.1-8B-Instruct)
-n, --max-samples   samples per domain (default: 128)
--batch-size        inference batch size (default: 1)
--seq-len           sequence length (default: 2048)
--top-k-pct         outlier threshold (default: 0.001)
--resume [TS]       resume latest or specific run
```

## Environment

Tested with Python 3.12, PyTorch 2.5.1, Transformers 4.46.3.
