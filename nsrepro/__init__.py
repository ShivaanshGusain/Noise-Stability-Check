"""Reproduction of the WikiText-2 experiment in Haris, Zhang & Yoshida, "Noise Stability of Transformer Models"
(arXiv 2602.08287, ICLR 2026), with and without two problems found in the released code:
a missing causal mask and a regularizer that takes the softmax over sequence positions instead of the vocabulary."""
