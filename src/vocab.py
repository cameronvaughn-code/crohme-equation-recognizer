"""
vocab.py

Builds and serializes the token vocabulary used by the decoder.
Special tokens:
  <pad>  - padding, ignored in loss
  <sos>  - start of sequence, fed as the decoder's first input
  <eos>  - end of sequence, decoding stops when this is produced
  <unk>  - token seen at inference/eval time that wasn't in the training vocab
"""

import json
from collections import Counter
from typing import Dict, List

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]


class Vocab:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

    def __len__(self):
        return len(self.token_to_id)

    def encode(self, tokens: List[str]) -> List[int]:
        unk_id = self.token_to_id[UNK]
        ids = [self.token_to_id[SOS]]
        ids += [self.token_to_id.get(t, unk_id) for t in tokens]
        ids += [self.token_to_id[EOS]]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> List[str]:
        tokens = [self.id_to_token.get(i, UNK) for i in ids]
        if strip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return tokens

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as f:
            token_to_id = json.load(f)
        return cls(token_to_id)

    @classmethod
    def build(cls, token_lists: List[List[str]], min_freq: int = 1) -> "Vocab":
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        vocab_tokens = list(SPECIAL_TOKENS)
        for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq:
                vocab_tokens.append(tok)
        token_to_id = {t: i for i, t in enumerate(vocab_tokens)}
        return cls(token_to_id)
