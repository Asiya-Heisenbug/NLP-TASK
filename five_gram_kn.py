from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class FiveGramKNModel:
    def __init__(self, unk_token: str = "<UNK>", discount: float = 0.75):
        self.unk_token = unk_token
        self.discount = discount
        self.vocab: set[str] = set()
        self.vocab_counter: Counter[str] = Counter()
        self.ngram_counts: Dict[Tuple[int, Tuple[str, ...]], int] = defaultdict(int)
        self._is_fitted = False

    def _normalize_token(self, token: str) -> str:
        return token if token in self.vocab else self.unk_token

    def _normalize_context(self, context: List[str]) -> List[str]:
        return [self._normalize_token(token) for token in context]

    def fit(self, corpus: List[List[str]]) -> None:
        self.vocab = {token for sentence in corpus for token in sentence}
        self.vocab.update({self.unk_token, "<s>", "</s>"})

        for sentence in corpus:
            tokens = [self._normalize_token(token) for token in sentence]
            padded = ["<s>"] * 4 + tokens + ["</s>"]
            for order in range(1, 6):
                for start in range(0, len(padded) - order + 1):
                    ngram = tuple(padded[start:start + order])
                    self.ngram_counts[(order, ngram)] += 1
                    if order == 1:
                        self.vocab_counter[ngram[0]] += 1

        self._is_fitted = True

    def probability(self, token: str, context: List[str]) -> float:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing probabilities")

        token = self._normalize_token(token)
        context = self._normalize_context(context)
        if len(context) > 4:
            context = context[-4:]

        if any(item not in self.vocab and item != self.unk_token for item in context):
            context = [self.unk_token if item not in self.vocab else item for item in context]

        if token == self.unk_token:
            return 1.0 / max(len(self.vocab), 1)

        if len(context) >= 4:
            history = tuple(context[-4:])
            ngram = history + (token,)
            context_count = self.ngram_counts[(4, history)]
            count = self.ngram_counts[(5, ngram)]
            if context_count and count:
                return max(count / context_count, 0.0)
            if len(history) > 1:
                return self.probability(token, list(history[1:]))
            return self._unigram_prob(token)

        if len(context) >= 3:
            history = tuple(context[-3:])
            ngram = history + (token,)
            context_count = self.ngram_counts[(3, history)]
            count = self.ngram_counts[(4, ngram)]
            if context_count and count:
                return max(count / context_count, 0.0)
            if len(history) > 1:
                return self.probability(token, list(history[1:]))
            return self._unigram_prob(token)

        if len(context) >= 2:
            history = tuple(context[-2:])
            ngram = history + (token,)
            context_count = self.ngram_counts[(2, history)]
            count = self.ngram_counts[(3, ngram)]
            if context_count and count:
                return max(count / context_count, 0.0)
            if len(history) > 1:
                return self.probability(token, list(history[1:]))
            return self._unigram_prob(token)

        if len(context) >= 1:
            history = tuple(context[-1:])
            ngram = history + (token,)
            context_count = self.ngram_counts[(1, history)]
            count = self.ngram_counts[(2, ngram)]
            if context_count and count:
                return max(count / context_count, 0.0)
            return self._unigram_prob(token)

        return self._unigram_prob(token)

    def _unigram_prob(self, token: str) -> float:
        total = sum(self.vocab_counter.values()) or 1
        return self.vocab_counter.get(token, 0) / total

    def suggest(self, prefix: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before suggesting")

        if any(token not in self.vocab for token in prefix):
            return [(self.unk_token, 1.0)]

        context = self._normalize_context(prefix)
        if len(context) > 4:
            context = context[-4:]

        candidates: List[Tuple[str, float]] = []
        for token in sorted(self.vocab):
            if token in {"<s>", "</s>", self.unk_token}:
                continue
            probability = self.probability(token, context)
            if probability > 0:
                candidates.append((token, probability))

        if not candidates:
            candidates.append((self.unk_token, 1.0))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]
