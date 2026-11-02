#!/usr/bin/env bash
# Step 1: does the released WikiText model only "learn" by reading its own label?
# Same model, data and settings, trained without (as released) and with a causal mask, 3 seeds each.
# If accuracy collapses with the causal mask, the paper's next-token result was measured on a model that could
# see the next token.
#
#   DATA=/path/to/wikitext OUT=runs bash scripts/leak_test.sh
set -euo pipefail
: "${DATA:?set DATA to the folder with wiki.train.tokens, wiki.valid.tokens, wiki.test.tokens}"
OUT="${OUT:-runs}"
EPOCHS="${EPOCHS:-60}"
SEEDS="${SEEDS:-0 1 2}"
PY="${PY:-python}"

for causal in 0 1; do
    "$PY" -m nsrepro.train --data-dir "$DATA" --out "$OUT/leak" --causal "$causal" --reg none \
        --epochs "$EPOCHS" --seeds $SEEDS
done
"$PY" -m nsrepro.summarize "$OUT/leak" --thresholds 0.3 0.5 0.7
