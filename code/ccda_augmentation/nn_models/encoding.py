TRIGGER_OPEN = "[T]"
TRIGGER_CLOSE = "[/T]"


def add_trigger_markers(tokenizer):
    specials = [t for t in (TRIGGER_OPEN, TRIGGER_CLOSE) if t not in tokenizer.get_vocab()]
    if specials:
        tokenizer.add_special_tokens({"additional_special_tokens": specials})
    return tokenizer


def char_encode(tokenizer, chars, trigger_span=None, max_len=256):
    tokens = [tokenizer.cls_token]
    char2tok = []
    for i, ch in enumerate(chars):
        if trigger_span is not None and i == trigger_span[0]:
            tokens.append(TRIGGER_OPEN)
        char2tok.append(len(tokens))
        sub = tokenizer.tokenize(ch)
        tokens.append(sub[0] if sub else tokenizer.unk_token)
        if trigger_span is not None and i == trigger_span[1] - 1:
            tokens.append(TRIGGER_CLOSE)
    tokens.append(tokenizer.sep_token)
    if len(tokens) > max_len:
        tokens = tokens[: max_len - 1] + [tokenizer.sep_token]
        char2tok = [t for t in char2tok if t < max_len - 1]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return input_ids, char2tok


def pad_batch(sequences, pad_id):
    max_len = max(len(s) for s in sequences)
    padded = []
    mask = []
    for s in sequences:
        padded.append(s + [pad_id] * (max_len - len(s)))
        mask.append([1] * len(s) + [0] * (max_len - len(s)))
    return padded, mask
