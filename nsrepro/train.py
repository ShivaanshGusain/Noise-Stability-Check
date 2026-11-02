"""Train the WikiText-2 next-token model of arXiv 2602.08287 (Sec. 6.1 / App. J.3) and log every epoch.

  python -m nsrepro.train --data-dir DIR --out runs/leak --causal 0 --reg none --seeds 0 1 2

Paper settings (App. J.3) are the defaults: 4 layers, width 30, 6 heads, sequence length 20, vocabulary 500,
batch 200, 1000 "training examples", weight decay 0.02. The released code also uses AdamW (lr 1e-3),
ReduceLROnPlateau on validation loss (patience 10, factor 0.5) and gradient clipping at 1.0.
Each run writes one JSON file with its settings and per-epoch metrics.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from nsrepro.data import UNK, load_wikitext, loaders
from nsrepro.model import SimpleTransformer
from nsrepro.regularizer import stability


@torch.no_grad()
def evaluate(model, loader, device, vocab_size, rho_track=0.5):
    """Loss, accuracy, accuracy on non-<UNK> targets, and noise stability at rho = 0.5 (fixed definition)."""
    model.eval()
    loss = correct = correct_known = known = total = 0.0
    stable, batches = 0.0, 0
    generator = torch.Generator(device=device).manual_seed(1234)  # same corruption every epoch
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        loss += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum").item()
        hit = logits.argmax(-1) == labels
        mask = labels != UNK
        correct += hit.sum().item()
        correct_known += (hit & mask).sum().item()
        known += mask.sum().item()
        total += labels.numel()
        stable += stability(model, ids, logits, rho_track, vocab_size, "fixed", generator).item()
        batches += 1
    model.train()
    return {"loss": loss / total, "acc": correct / total, "acc_known": correct_known / max(known, 1),
            "stability": stable / max(batches, 1)}


def run(args, seed, tokens, vocab_size, device):
    torch.manual_seed(seed)
    model = SimpleTransformer(vocab_size, args.width, args.layers, args.heads, args.seq_length, args.dropout,
                              causal=bool(args.causal)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=args.patience,
                                                            factor=args.lr_factor) if args.patience else None)
    train, evaluation = loaders(tokens, args.seq_length, args.batch_size, seed)
    generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    history, steps, forwards, began = [], 0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        task_total, reg_total = 0.0, 0.0
        for ids, labels in train:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            task = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss = task
            forwards += 1
            if args.reg != "none" and args.gamma > 0:
                stable = stability(model, ids, logits, args.rho, vocab_size, args.reg, generator)
                loss = task - args.gamma * stable
                reg_total += stable.item()
                forwards += 2 if args.reg == "original" else 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            task_total += task.item()
            steps += 1
        val = evaluate(model, evaluation["val"], device, vocab_size)
        if scheduler is not None:
            scheduler.step(val["loss"])
        history.append({"epoch": epoch, "steps": steps, "forward_passes": forwards,
                        "seconds": round(time.time() - began, 2), "train_loss": task_total / len(train),
                        "train_stability": reg_total / len(train) if args.reg != "none" else None,
                        "lr": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val.items()}})
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"seed {seed} epoch {epoch} step {steps}: val loss {val['loss']:.3f}, acc {val['acc']:.3f}, "
                  f"acc(non-UNK) {val['acc_known']:.3f}, stability {val['stability']:.3f}", flush=True)
    test = evaluate(model, evaluation["test"], device, vocab_size)
    return {"seed": seed, "history": history, "test": test}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Folder with wiki.train/valid/test.tokens")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", help="Run name (default: built from the settings)")
    parser.add_argument("--causal", type=int, choices=[0, 1], default=1, help="0 = as released (leaks the label)")
    parser.add_argument("--reg", choices=["none", "original", "fixed"], default="none")
    parser.add_argument("--gamma", type=float, default=0.0, help="Regularization strength")
    parser.add_argument("--rho", type=float, default=0.05, help="Correlation; tokens replaced w.p. (1 - rho) / 2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=10, help="ReduceLROnPlateau patience (0 = no scheduler)")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args(argv)
    if args.reg != "none" and args.gamma <= 0:
        parser.error("--reg needs --gamma > 0")

    tokenizer, tokens = load_wikitext(args.data_dir, args.vocab_size, args.seq_length, args.train_samples)
    vocab_size = len(tokenizer)
    label = args.label or (f"causal{args.causal}_{args.reg}" +
                           (f"_g{args.gamma:g}_r{args.rho:g}" if args.reg != "none" else ""))
    unk = {split: sum(t == UNK for t in toks) / max(len(toks), 1) for split, toks in tokens.items()}
    print(f"{label}: vocabulary {vocab_size}, tokens {({k: len(v) for k, v in tokens.items()})}, "
          f"<UNK> share {({k: round(v, 3) for k, v in unk.items()})}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    for seed in args.seeds:
        path = os.path.join(args.out, f"{label}_seed{seed}.json")
        if os.path.exists(path):
            print(f"{path} exists, skipping")
            continue
        result = run(args, seed, tokens, vocab_size, args.device)
        result.update(label=label, settings=vars(args), unk_share=unk)
        with open(path + ".tmp", "w") as handle:
            json.dump(result, handle)
        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
