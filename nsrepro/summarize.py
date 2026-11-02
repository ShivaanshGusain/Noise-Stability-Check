"""Summarize runs: per setting, mean ± std over seeds of the best validation accuracy (all and non-<UNK>), the best
validation loss, and the training steps / forward passes needed to first reach each accuracy threshold.

  python -m nsrepro.summarize runs/leak runs/repro --thresholds 0.3 0.5 0.7
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def first_reach(history, key, threshold):
    for row in history:
        if row[key] >= threshold:
            return row["steps"], row["forward_passes"]
    return None, None


def fmt(values, digits=3):
    values = [v for v in values if v is not None]
    if not values:
        return "never"
    if len(values) == 1:
        return f"{values[0]:.{digits}g}"
    return f"{statistics.mean(values):.{digits}g} ± {statistics.stdev(values):.{digits - 1}g}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--key", default="val_acc", choices=["val_acc", "val_acc_known"])
    args = parser.parse_args(argv)
    groups = defaultdict(list)
    for folder in args.folders:
        for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
            with open(path) as handle:
                run = json.load(handle)
            groups[(folder, run["label"])].append(run)
    if not groups:
        print(f"No finished runs (*.json) found in {args.folders}. Run scripts/leak_test.sh first.")
        return
    for (folder, label), runs in groups.items():
        print(f"\n{folder} :: {label}  ({len(runs)} seeds)")
        h = [r["history"] for r in runs]
        print(f"  best val acc          {fmt([max(x['val_acc'] for x in hist) for hist in h])}")
        print(f"  best val acc (non-UNK) {fmt([max(x['val_acc_known'] for x in hist) for hist in h])}")
        print(f"  best val loss         {fmt([min(x['val_loss'] for x in hist) for hist in h])}")
        print(f"  test acc / non-UNK    {fmt([r['test']['acc'] for r in runs])} / "
              f"{fmt([r['test']['acc_known'] for r in runs])}")
        for threshold in args.thresholds:
            reached = [first_reach(hist, args.key, threshold) for hist in h]
            steps = [s for s, _ in reached]
            passes = [p for _, p in reached]
            hit = sum(s is not None for s in steps)
            print(f"  {args.key} >= {threshold:g}: reached by {hit}/{len(runs)} seeds; steps {fmt(steps)}; "
                  f"forward passes {fmt(passes)}")


if __name__ == "__main__":
    main()
