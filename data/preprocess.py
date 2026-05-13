def tokenize_texts(tokenizer, texts, seq_len=2048):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=seq_len,
        return_tensors="pt",
    ).input_ids
