from datasets import load_dataset

DOMAIN_CONFIG = {
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "train",
        "text_key": "text",
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "text_key": "question",
    },
    "code_alpaca": {
        "path": "flwrlabs/code-alpaca-20k",
        "name": None,
        "split": "train",
        "text_key": None,
    },
    "pubmed": {
        "path": "slinusc/PubMedAbstractsSubset",
        "name": None,
        "split": "train",
        "text_key": "abstract",
    },
}


def load_domain(domain, max_samples=128):
    cfg = DOMAIN_CONFIG[domain]
    kwargs = {"split": cfg["split"], "streaming": True}
    if cfg["name"]:
        kwargs["name"] = cfg["name"]

    ds = load_dataset(cfg["path"], **kwargs)

    texts = []
    for item in ds:
        if max_samples and len(texts) >= max_samples:
            break
        if domain == "code_alpaca":
            parts = [item["instruction"]]
            inp = item.get("input", "").strip()
            if inp:
                parts.append(inp)
            parts.append(item["output"])
            text = "\n".join(parts)
        else:
            text = item.get(cfg["text_key"], "")
            if not text or not text.strip():
                continue
        texts.append(text)

    return texts
