"""
metrics.py

Evaluation metrics for image-to-LaTeX.

Exact match is the headline CROHME metric (the whole predicted expression
must be token-for-token identical to the reference), but it's brutal early
in training, so we also track token-level accuracy via edit distance:

  - token_error_rate: Levenshtein distance between predicted and reference
    token sequences, divided by reference length (like WER for speech).
  - token_accuracy: 1 - token_error_rate, clipped to [0, 1].

All functions operate on lists of tokens (special tokens already stripped).
"""

from typing import List, Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Standard edit distance (insert / delete / substitute, unit cost)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = curr
    return prev[m]


class MetricAccumulator:
    """Aggregates exact-match and token-error-rate over a dataset."""

    def __init__(self) -> None:
        self.n = 0
        self.exact = 0
        self.total_edits = 0
        self.total_ref_tokens = 0

    def update(self, pred_tokens: List[str], ref_tokens: List[str]) -> None:
        self.n += 1
        if pred_tokens == ref_tokens:
            self.exact += 1
        self.total_edits += levenshtein(pred_tokens, ref_tokens)
        self.total_ref_tokens += max(len(ref_tokens), 1)

    @property
    def exact_match(self) -> float:
        return self.exact / max(self.n, 1)

    @property
    def token_error_rate(self) -> float:
        return self.total_edits / max(self.total_ref_tokens, 1)

    @property
    def token_accuracy(self) -> float:
        return max(0.0, 1.0 - self.token_error_rate)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "exact_match": round(self.exact_match, 4),
            "token_error_rate": round(self.token_error_rate, 4),
            "token_accuracy": round(self.token_accuracy, 4),
        }
