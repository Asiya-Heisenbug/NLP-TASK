# Sparse 5-gram Language Model with Kneser-Ney Smoothing

This project implements a 5-gram language model from scratch for real-time autocomplete and query prediction. It uses:

- sparse n-gram counting over observed sequences
- Kneser-Ney-style discounting
- recursive back-off for unseen prefixes
- dynamic handling of OOV tokens via a dedicated `<UNK>` symbol

## Files

- `five_gram_kn.py`: core model implementation
- `tests/test_five_gram_kn.py`: unit tests for fitting, scoring, suggestion, and OOV behavior

## Usage

```python
from five_gram_kn import FiveGramKNModel

model = FiveGramKNModel()
corpus = [
    ["best", "places", "to", "visit", "in", "india"],
    ["best", "places", "to", "visit", "near", "me"],
]
model.fit(corpus)

suggestions = model.suggest(["best", "places", "to", "visit"], top_k=4)
print(suggestions)
```

## Testing

Run:

```bash
python -m pytest -q
```
