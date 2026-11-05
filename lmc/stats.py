"""Paired and one-sample t-tests with Benjamini–Hochberg false-discovery-rate control."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TTestResult:
    """Result of a paired or one-sample t-test on differences d = x - y."""

    n: int
    mean_diff: float     # mean of (x - y)
    sd_diff: float       # sample standard deviation of (x - y)
    t: float
    p: float             # two-sided
    q: float = math.nan  # Benjamini–Hochberg adjusted p, set by `benjamini_hochberg`

    @property
    def sem(self) -> float:
        return self.sd_diff / math.sqrt(self.n)


def paired_ttest(x, y) -> TTestResult:
    """Two-sided paired t-test of mean(x - y) = 0. x[i] and y[i] must come from the same seed."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    sd = d.std(ddof=1)
    if sd == 0:
        return TTestResult(len(d), float(d.mean()), 0.0, math.nan, math.nan)
    t, p = stats.ttest_rel(x, y)
    return TTestResult(len(d), float(d.mean()), float(sd), float(t), float(p))


def one_sample_ttest(values) -> TTestResult:
    """Two-sided one-sample t-test of mean(values) = 0."""
    return paired_ttest(values, np.zeros(len(values)))


def benjamini_hochberg(tests: list[TTestResult]) -> list[TTestResult]:
    """Set .q on every test with a finite p, treating the list as one family. Modifies in place."""
    idx = [i for i, r in enumerate(tests) if not math.isnan(r.p)]
    if idx:
        q = stats.false_discovery_control([tests[i].p for i in idx], method="bh")
        for i, qi in zip(idx, q):
            tests[i].q = float(qi)
    return tests


def mean_sem(values) -> tuple[float, float]:
    """Mean and standard error of the non-None values (NaN when undefined)."""
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) == 0:
        return math.nan, math.nan
    sem = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else math.nan
    return float(v.mean()), sem
