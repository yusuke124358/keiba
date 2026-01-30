import math
import random


def test_pl_top3_order_probs_sum_to_one():
    from keiba.exotics.probability import pl_all_top3_order_probs, sum_probs

    random.seed(42)
    n = 10
    raw = [random.random() for _ in range(n)]
    s = sum(raw)
    p = {i + 1: raw[i] / s for i in range(n)}  # 馬番は1..n

    probs = pl_all_top3_order_probs(p)
    total = sum_probs(probs)
    assert math.isfinite(total)
    assert abs(total - 1.0) < 1e-9


def test_pl_trio_set_probs_sum_to_one():
    from keiba.exotics.probability import pl_all_trio_set_probs, sum_probs

    random.seed(123)
    n = 12
    raw = [random.random() for _ in range(n)]
    s = sum(raw)
    p = {i + 1: raw[i] / s for i in range(n)}

    probs = pl_all_trio_set_probs(p)
    total = sum_probs(probs)
    assert math.isfinite(total)
    assert abs(total - 1.0) < 1e-9


