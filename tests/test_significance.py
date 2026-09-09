"""Pin metrics/significance.py against scipy and statsmodels.

The module is stdlib-only so the report can be regenerated on a shared host.
That is only defensible if the hand-rolled distribution functions are provably
identical to the reference implementations, which is what this checks.

    pip install -r requirements-dev.txt
    python tests/test_significance.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from metrics import significance as S  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.contingency_tables import cochrans_q as sm_cochrans_q  # noqa: E402
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

failures = []


def close(label, got, want, tol=1e-9):
    if got is None or want is None:
        ok = got is want
    else:
        ok = abs(got - want) <= tol * max(1.0, abs(want))
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r} vs {want!r}")
    if not ok:
        failures.append(label)


def test_chi2_sf():
    print("\nchi2_sf vs scipy.stats.chi2.sf")
    for df in (1, 2, 3, 5, 10, 57):
        for x in (0.001, 0.5, 1.0, 3.84, 12.5, 60.0, 200.0):
            close(f"df={df} x={x}", S.chi2_sf(x, df), float(stats.chi2.sf(x, df)), 1e-10)


def test_binom():
    print("\nbinom_two_sided_p vs scipy.stats.binomtest")
    for n, k in ((10, 3), (25, 12), (40, 5), (100, 61), (7, 0), (50, 25)):
        want = float(stats.binomtest(k, n, 0.5).pvalue)
        close(f"n={n} k={k}", S.binom_two_sided_p(k, n), want, 1e-12)


def test_mcnemar():
    print("\nmcnemar vs statsmodels.stats.contingency_tables.mcnemar")
    rng = random.Random(11)
    for trial, (n, pa, pb) in enumerate(
            [(2850, 0.79, 0.60), (1000, 0.22, 0.18), (300, 0.5, 0.5), (60, 0.4, 0.42)]):
        a = [rng.random() < pa for _ in range(n)]
        b = [rng.random() < pb for _ in range(n)]
        got = S.mcnemar(a, b)

        n01 = sum(1 for x, y in zip(a, b) if x and not y)
        n10 = sum(1 for x, y in zip(a, b) if not x and y)
        table = np.array([[sum(1 for x, y in zip(a, b) if x and y), n01],
                          [n10, sum(1 for x, y in zip(a, b) if not x and not y)]])
        exact = got["discordant"] < 25
        want = sm_mcnemar(table, exact=exact, correction=True)
        close(f"trial {trial} p", got["p_value"], float(want.pvalue), 1e-9)
        if not exact:
            close(f"trial {trial} stat", got["statistic"],
                  round(float(want.statistic), 4), 1e-6)


def test_cochran():
    print("\ncochrans_q vs statsmodels")
    rng = random.Random(7)
    n = 800
    models = {f"m{i}": [rng.random() < p for _ in range(n)]
              for i, p in enumerate((0.75, 0.60, 0.58))}
    got = S.cochrans_q(models)
    arr = np.array([[int(models[m][i]) for m in models] for i in range(n)])
    want = sm_cochrans_q(arr)
    close("Q", got["q_statistic"], round(float(want.statistic), 3), 1e-6)
    close("p", got["p_value"], float(want.pvalue), 1e-9)


def test_chi2_independence():
    print("\nchi2_independence vs scipy.stats.chi2_contingency")
    rng = random.Random(3)
    groups = {g: [rng.random() < p for _ in range(n)]
              for g, p, n in (("a", 0.7, 400), ("b", 0.5, 350), ("c", 0.55, 300))}
    got = S.chi2_independence(groups)
    obs = np.array([[sum(1 for v in groups[g] if v), sum(1 for v in groups[g] if not v)]
                    for g in groups])
    chi2, p, dof, _ = stats.chi2_contingency(obs, correction=False)
    close("chi2", got["chi2"], round(float(chi2), 3), 1e-6)
    close("p", got["p_value"], float(p), 1e-9)
    close("df", got["df"], int(dof))


def test_holm():
    print("\nholm_bonferroni vs statsmodels.stats.multitest")
    for ps in ([0.01, 0.04, 0.03], [1e-9, 0.2, 0.049], [0.5, 0.5, 0.5],
               [0.001, 0.002, 0.003, 0.9]):
        got = S.holm_bonferroni(ps)
        _, want, _, _ = multipletests(ps, alpha=0.05, method="holm")
        for i, (g, w) in enumerate(zip(got, want)):
            close(f"{ps} [{i}]", g["p_adjusted"], float(w), 1e-12)


def test_variance_decomposition():
    """No reference implementation, so check invariants over many seeds.

    Single-seed thresholds would be flaky: the estimator is
    max(0, observed - within), and that truncation biases `between_share`
    upward when a single draw happens to land above the noise floor. Averaging
    over seeds is what makes the assertion meaningful.
    """
    print("\nvariance_decomposition invariants")

    # All groups share one true rate, so every bit of spread is sampling noise
    # and the between-group share must average near zero.
    shares = []
    for seed in range(30):
        rng = random.Random(seed)
        noise = {f"s{i}": [rng.random() < 0.6 for _ in range(50)] for i in range(57)}
        shares.append(S.variance_decomposition(noise)["between_share"])
    mean_share = sum(shares) / len(shares)
    ok = mean_share < 0.15
    print(f"  {'ok  ' if ok else 'FAIL'} identical rates -> mean between_share "
          f"{mean_share:.4f} over 30 seeds (want < 0.15, max {max(shares):.3f})")
    if not ok:
        failures.append("variance: noise floor")

    # Genuinely different rates must be recovered as real signal, every time.
    worst = 1.0
    for seed in range(30):
        rng = random.Random(100 + seed)
        spread = {f"s{i}": [rng.random() < (0.2 + 0.011 * i) for _ in range(50)]
                  for i in range(57)}
        worst = min(worst, S.variance_decomposition(spread)["between_share"])
    ok = worst > 0.85
    print(f"  {'ok  ' if ok else 'FAIL'} spread rates -> worst between_share "
          f"{worst:.4f} over 30 seeds (want > 0.85)")
    if not ok:
        failures.append("variance: signal recovery")

    # Identity holds to the precision the values are rounded to (6 dp).
    rng = random.Random(5)
    sample = {f"s{i}": [rng.random() < 0.6 for _ in range(50)] for i in range(57)}
    got = S.variance_decomposition(sample)
    ident = abs(got["between_group_variance"]
                - max(0.0, got["observed_variance"] - got["sampling_variance"])) < 1e-6
    print(f"  {'ok  ' if ident else 'FAIL'} observed = between + sampling "
          f"(to the reported 6 dp)")
    if not ident:
        failures.append("variance: identity")


def main():
    test_chi2_sf()
    test_binom()
    test_mcnemar()
    test_cochran()
    test_chi2_independence()
    test_holm()
    test_variance_decomposition()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
    else:
        print("all statistics match the reference implementations")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
