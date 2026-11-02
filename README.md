# noise-stability-repro

Reproduction of the next-token (WikiText-2) experiment in Haris, Zhang & Yoshida, *Noise Stability of Transformer
Models* (arXiv 2602.08287, ICLR 2026, Sec. 6.1 and App. J.3). The experiment is run as released and with two
problems found in the released code (supplementary `Transformers_Sensitivity/`) fixed.

## Problems in the released code

1. **No causal mask** (`model/transformer.py`, attention). The only mask is a padding mask, and it is all ones for
   WikiText. Labels are the input shifted by one (`utilities/wikitext_data.py`), so position *i* can attend to
   position *i + 1*, which is its own label. "Next-token" accuracy can then come from copying.
2. **The regularizer takes the softmax over positions** (`noise_stability/measure_noise_stability.py`):
   `F.softmax(outputs, dim=1)` on outputs of shape `[batch, length, vocab]`. Definition 6.1 needs next-token
   distributions (softmax over the vocabulary). For the classification tasks (`[batch, classes]`), `dim=1` is
   correct, so those results are not affected.
3. **Minor: two extra forward passes.** The regularizer recomputes the clean forward pass, so it costs two extra
   passes per step, not the one the paper states.

Also: "1000 training examples" are the first 1000 × 21 tokens cut into windows at every offset, which gives
about 21,000 overlapping windows. With a 500-word vocabulary much of the text is `<UNK>`. All runs therefore also
report accuracy on targets that are not `<UNK>`.

## What runs

| Script | Question | Runs |
|---|---|---|
| `scripts/leak_test.sh` | Does accuracy collapse once the causal mask is added? | 2 settings × 3 seeds |
| `scripts/repro_sweep.sh` | (A) Does this reimplementation reproduce the paper's ~75% speed-up as released? (B) Does it survive the fixes? | (1 + 6) × 2 settings × 5 seeds |

Each run is a 4-layer, width-30 model (paper settings). Runs are small, but timings on the GX10 have not been
measured; the log prints every 5 epochs.

## Setup on the GX10

```bash
# copy this folder and the three WikiText files to the pod, then:
cd noise-stability-repro
source /path/to/your/venv/bin/activate     # needs only torch (and pytest for the tests)
python -m pytest -q tests                  # CPU, under a minute
DATA=/path/to/wikitext OUT=runs bash scripts/leak_test.sh
DATA=/path/to/wikitext OUT=runs bash scripts/repro_sweep.sh
```

`DATA` is the folder holding `wiki.train.tokens`, `wiki.valid.tokens` and `wiki.test.tokens`. They are in the
supplementary material at `Transformers_Sensitivity/data/wikitext/`. Finished runs are skipped when a script is
re-run. Results are JSON files under `runs/`; `python -m nsrepro.summarize runs/leak` prints the summary again.

## Reading the results

- **Leak test.** If `causal0` reaches high accuracy (the paper reports about 70%) and `causal1` stays far lower
  with the same settings, the released next-token result relied on seeing the label.
- **Sweep A (as released).** It should reproduce the paper's pattern: with the original regularizer, the 70%
  threshold is reached in clearly fewer steps. If it does not, this reimplementation differs from what produced
  Figure 6, and that must be resolved before anything else.
- **Sweep B (corrected).** The real question is whether the fixed regularizer reaches a given accuracy in clearly
  fewer steps across seeds. Compare **forward passes** too, since the regularizer costs one extra pass per step.
  Kill the "regularizer speeds up language modelling" line if, at equal forward passes, no (γ, ρ) beats the
  baseline by more than the seed spread.
