import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nsrepro.data import UNK, Tokenizer, Windows, words  # noqa: E402
from nsrepro.model import SimpleTransformer  # noqa: E402
from nsrepro.regularizer import corrupt, stability  # noqa: E402
from nsrepro.summarize import first_reach  # noqa: E402
from nsrepro.train import main  # noqa: E402

VOCAB = 40


def tiny(causal):
    torch.manual_seed(0)
    return SimpleTransformer(VOCAB, width=12, layers=2, heads=3, max_length=10, causal=causal).eval()


def test_causal_mask_hides_the_future_and_the_released_model_does_not():
    ids = torch.randint(3, VOCAB, (2, 10), generator=torch.Generator().manual_seed(1))
    changed = ids.clone()
    changed[:, 6] = (changed[:, 6] + 1) % VOCAB
    for causal, expect_same in ((True, True), (False, False)):
        model = tiny(causal)
        before, after = model(ids), model(changed)
        same = torch.allclose(before[:, :6], after[:, :6], atol=1e-6)
        assert same == expect_same  # with the mask, positions before 6 cannot see token 6
        assert not torch.allclose(before[:, 6:], after[:, 6:])


def test_fixed_regularizer_is_a_probability_overlap_over_the_vocabulary():
    model = tiny(True)
    ids = torch.randint(0, VOCAB, (3, 10), generator=torch.Generator().manual_seed(2))
    logits = model(ids)
    value = stability(model, ids, logits, 1.0, VOCAB, "fixed")  # rho = 1: nothing is replaced
    expected = F_softmax_overlap(logits, logits)
    torch.testing.assert_close(value, expected)
    assert 0 < value.item() <= 1
    value.backward()  # differentiable
    assert any(p.grad is not None for p in model.parameters())


def F_softmax_overlap(a, b):
    return (torch.softmax(a, -1) * torch.softmax(b, -1)).sum(-1).mean()


def test_original_regularizer_softmaxes_over_positions_as_released():
    model = tiny(False)
    ids = torch.randint(0, VOCAB, (3, 10), generator=torch.Generator().manual_seed(3))
    logits = model(ids)
    value = stability(model, ids, logits, 1.0, VOCAB, "original")
    p = torch.softmax(logits, dim=1)
    torch.testing.assert_close(value, (p * p).sum(dim=1).mean())


def test_corruption_rate_matches_one_minus_rho_over_two():
    ids = torch.full((200, 200), 7)
    noisy = corrupt(ids, 0.5, 10_000, torch.Generator().manual_seed(4))
    rate = (noisy != ids).float().mean().item()
    assert abs(rate - 0.25) < 0.01  # replaced w.p. (1 - 0.5) / 2; a random draw equal to 7 is negligible


def test_tokenizer_and_windows_follow_the_released_code():
    assert words("Hello, World! It's 3.") == ["hello", ",", "world", "!", "it", "s", "3", "."]
    tok = Tokenizer("a a a b b c", vocab_size=5)
    assert tok.encode("a b c z") == [3, 4, UNK, UNK]  # only vocab_size - 3 = 2 words kept
    windows = Windows(list(range(10)), 4)
    assert len(windows) == 6  # every offset (stride 1)
    x, y = windows[2]
    assert x.tolist() == [2, 3, 4, 5] and y.tolist() == [3, 4, 5, 6]


def test_first_reach():
    history = [{"steps": 10, "forward_passes": 20, "val_acc": 0.2}, {"steps": 20, "forward_passes": 40, "val_acc": 0.6}]
    assert first_reach(history, "val_acc", 0.5) == (20, 40)
    assert first_reach(history, "val_acc", 0.9) == (None, None)


def test_training_cli_end_to_end(tmp_path):
    data = tmp_path / "wikitext"
    data.mkdir()
    text = " = Title = \n" + "the cat sat on the mat . the dog sat on the log . " * 60 + "\n"
    for name in ("train", "valid", "test"):
        (data / f"wiki.{name}.tokens").write_text(text)
    out = tmp_path / "runs"
    common = ["--data-dir", str(data), "--out", str(out), "--epochs", "2", "--seeds", "0", "--device", "cpu",
              "--width", "12", "--heads", "3", "--layers", "2", "--vocab-size", "20", "--train-samples", "20",
              "--batch-size", "16", "--log-every", "1"]
    main([*common, "--causal", "1", "--reg", "fixed", "--gamma", "0.5", "--rho", "0.5"])
    main([*common, "--causal", "0", "--reg", "original", "--gamma", "0.5"])
    files = sorted(p.name for p in out.iterdir())
    assert files == ["causal0_original_g0.5_r0.05_seed0.json", "causal1_fixed_g0.5_r0.5_seed0.json"]
