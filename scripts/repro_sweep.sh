#!/usr/bin/env bash
# Step 2: does the regularizer's "~75% faster" next-token result survive the fixes?
#   A. As released (no causal mask, original regularizer): checks that this reimplementation reproduces the paper.
#   B. Corrected (causal mask, regularizer over the vocabulary): the real test.
# Several (gamma, rho) settings, as the paper says the best one needs a sweep; 5 seeds each.
#
#   DATA=/path/to/wikitext OUT=runs bash scripts/repro_sweep.sh
set -euo pipefail
: "${DATA:?set DATA to the folder with wiki.train.tokens, wiki.valid.tokens, wiki.test.tokens}"
OUT="${OUT:-runs}"
EPOCHS="${EPOCHS:-60}"
SEEDS="${SEEDS:-0 1 2 3 4}"
GAMMAS="${GAMMAS:-0.1 0.5 1.0}"
RHOS="${RHOS:-0.05 0.5}"
PY="${PY:-python}"

common=(--data-dir "$DATA" --epochs "$EPOCHS" --seeds $SEEDS)
# A. As released.
"$PY" -m nsrepro.train "${common[@]}" --out "$OUT/released" --causal 0 --reg none
for g in $GAMMAS; do for r in $RHOS; do
    "$PY" -m nsrepro.train "${common[@]}" --out "$OUT/released" --causal 0 --reg original --gamma "$g" --rho "$r"
done; done
# B. Corrected.
"$PY" -m nsrepro.train "${common[@]}" --out "$OUT/corrected" --causal 1 --reg none
for g in $GAMMAS; do for r in $RHOS; do
    "$PY" -m nsrepro.train "${common[@]}" --out "$OUT/corrected" --causal 1 --reg fixed --gamma "$g" --rho "$r"
done; done

"$PY" -m nsrepro.summarize "$OUT/released" --thresholds 0.3 0.5 0.7
"$PY" -m nsrepro.summarize "$OUT/corrected" --thresholds 0.1 0.2 0.3
"$PY" -m nsrepro.summarize "$OUT/corrected" --key val_acc_known --thresholds 0.1 0.2 0.3
