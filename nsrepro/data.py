"""WikiText-2 exactly as the released code prepares it (utilities/wikitext_data.py).

- Lines starting with '=' (section headers) and empty lines are dropped; the rest are joined with spaces.
- Word tokenizer: lowercase, regex r'\\b\\w+\\b|[.,!?;]', vocabulary = <PAD>, <UNK>, <EOS> + the most common
  (vocab_size - 3) training words.
- The training split keeps only the first train_samples * (seq_length + 1) tokens; validation and test keep
  200 * (seq_length + 1). Windows are taken at EVERY offset (stride 1), so "1000 training examples" means about
  21,000 overlapping windows for seq_length 20.
"""

import os
import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

PAD, UNK, EOS = 0, 1, 2
WORDS = re.compile(r"\b\w+\b|[.,!?;]")


def load_split(path):
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.read().split("\n")]
    return " ".join(line for line in lines if line and not line.startswith("="))


def words(text):
    return WORDS.findall(text.lower())


class Tokenizer:
    def __init__(self, train_text, vocab_size):
        counts = Counter(words(train_text))
        self.word2idx = {"<PAD>": PAD, "<UNK>": UNK, "<EOS>": EOS}
        for index, (word, _) in enumerate(counts.most_common(vocab_size - 3), start=3):
            self.word2idx[word] = index

    def encode(self, text):
        return [self.word2idx.get(word, UNK) for word in words(text)]

    def __len__(self):
        return len(self.word2idx)


class Windows(Dataset):
    """Every window of seq_length tokens (stride 1); labels are the same window shifted by one token."""

    def __init__(self, tokens, seq_length):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_length)

    def __getitem__(self, index):
        window = self.tokens[index:index + self.seq_length + 1]
        return window[:-1], window[1:]


def load_wikitext(data_dir, vocab_size=500, seq_length=20, train_samples=1000, eval_samples=200):
    """Returns (tokenizer, {"train", "val", "test"} token lists)."""
    texts = {split: load_split(os.path.join(data_dir, f"wiki.{name}.tokens"))
             for split, name in (("train", "train"), ("val", "valid"), ("test", "test"))}
    tokenizer = Tokenizer(texts["train"], vocab_size)
    limits = {"train": train_samples, "val": eval_samples, "test": eval_samples}
    tokens = {split: tokenizer.encode(text)[:limits[split] * (seq_length + 1)] for split, text in texts.items()}
    return tokenizer, tokens


def loaders(tokens, seq_length, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(Windows(tokens["train"], seq_length), batch_size=batch_size, shuffle=True,
                       generator=generator)
    evaluation = {split: DataLoader(Windows(tokens[split], seq_length), batch_size=batch_size)
                  for split in ("val", "test")}
    return train, evaluation
