"""Statistical analysis utilities for evaluation framework."""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None
    is_significant: bool
    confidence_level: float
    mean_diff: float
    ci_lower: float
    ci_upper: float
    sample_size_a: int
    sample_size_b: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "mean_diff": self.mean_diff,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "sample_size_a": self.sample_size_a,
            "sample_size_b": self.sample_size_b,
        }


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis."""

    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_bootstrap: int
    confidence_level: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mean_diff": self.mean_diff,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
        }


@dataclass
class BootstrapMAPResult:
    """Result of bootstrap CI on a single MAP estimate."""

    mean: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    n_samples: int
    n_bootstrap: int
    confidence_level: float

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_width": self.ci_width,
            "n_samples": self.n_samples,
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
        }


def bootstrap_map_ci(
    ap_values: Sequence[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapMAPResult:
    """Bootstrap CI on MAP (mean of per-sample APs).

    Captures test-set composition uncertainty, NOT seed variability.
    """
    import random

    rng = random.Random(seed)
    values = list(ap_values)
    n = len(values)

    if n == 0:
        return BootstrapMAPResult(
            mean=float("nan"), ci_lower=float("nan"), ci_upper=float("nan"),
            ci_width=float("nan"), n_samples=0, n_bootstrap=n_bootstrap,
            confidence_level=confidence,
        )

    observed_mean = sum(values) / n
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choices(values, k=n)
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = (1 - confidence) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[int((1 - alpha) * n_bootstrap)]

    return BootstrapMAPResult(
        mean=observed_mean, ci_lower=lo, ci_upper=hi, ci_width=hi - lo,
        n_samples=n, n_bootstrap=n_bootstrap, confidence_level=confidence,
    )


def compute_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval for a sequence of values.

    Args:
        values: Sequence of values.
        confidence: Confidence level (default: 0.95).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    import statistics

    if not values or len(values) < 2:
        mean = values[0] if values else float("nan")
        return (mean, mean)

    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values)

    try:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    except ImportError:
        # Approximate t-value for common confidence levels
        t_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        t_value = t_values.get(confidence, 1.96)

    margin = t_value * (std / (n ** 0.5))
    return (mean - margin, mean + margin)


def compute_cohens_d(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> float:
    """Compute Cohen's d effect size.

    Args:
        values_a: First group values.
        values_b: Second group values.

    Returns:
        Cohen's d effect size.
    """
    import statistics

    if not values_a or not values_b:
        return float("nan")

    mean_a = statistics.mean(values_a)
    mean_b = statistics.mean(values_b)

    n_a = len(values_a)
    n_b = len(values_b)

    var_a = statistics.variance(values_a) if n_a > 1 else 0
    var_b = statistics.variance(values_b) if n_b > 1 else 0

    # Pooled standard deviation
    pooled_std = ((var_a * (n_a - 1) + var_b * (n_b - 1)) / (n_a + n_b - 2)) ** 0.5

    if pooled_std == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_std


def compare_models_paired_ttest(
    ap_values_a: Sequence[float],
    ap_values_b: Sequence[float],
    confidence: float = 0.95,
) -> StatisticalTestResult:
    """Compare two models using paired t-test.

    Requires paired samples (same test set, different models).

    Args:
        ap_values_a: AP values for model A.
        ap_values_b: AP values for model B.
        confidence: Confidence level for significance.

    Returns:
        StatisticalTestResult with test results.
    """
    import statistics

    if len(ap_values_a) != len(ap_values_b):
        raise ValueError("Paired t-test requires equal sample sizes")

    n = len(ap_values_a)
    if n < 2:
        raise ValueError("Need at least 2 samples for t-test")

    # Compute differences
    diffs = [a - b for a, b in zip(ap_values_a, ap_values_b)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)

    # t-statistic
    t_stat = mean_diff / (std_diff / (n ** 0.5))

    try:
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    except ImportError:
        # Rough approximation
        p_value = 2 * (1 - 0.5 * (1 + (abs(t_stat) / ((n - 1) ** 0.5))))
        t_crit = 1.96

    margin = t_crit * (std_diff / (n ** 0.5))
    ci_lower = mean_diff - margin
    ci_upper = mean_diff + margin

    effect_size = compute_cohens_d(ap_values_a, ap_values_b)
    is_significant = p_value < (1 - confidence)

    return StatisticalTestResult(
        test_name="paired_ttest",
        statistic=t_stat,
        p_value=p_value,
        effect_size=effect_size,
        is_significant=is_significant,
        confidence_level=confidence,
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        sample_size_a=n,
        sample_size_b=n,
    )


def compare_models_bootstrap(
    ap_values_a: Sequence[float],
    ap_values_b: Sequence[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compare two models using bootstrap resampling.

    Does not require paired samples.

    Args:
        ap_values_a: AP values for model A.
        ap_values_b: AP values for model B.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with analysis results.
    """
    import random
    import statistics

    rng = random.Random(seed)

    mean_a = statistics.mean(ap_values_a)
    mean_b = statistics.mean(ap_values_b)
    observed_diff = mean_a - mean_b

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choices(list(ap_values_a), k=len(ap_values_a))
        sample_b = rng.choices(list(ap_values_b), k=len(ap_values_b))
        diff = statistics.mean(sample_a) - statistics.mean(sample_b)
        bootstrap_diffs.append(diff)

    # Compute CI
    alpha = (1 - confidence) / 2
    bootstrap_diffs.sort()
    lower_idx = int(alpha * n_bootstrap)
    upper_idx = int((1 - alpha) * n_bootstrap)
    ci_lower = bootstrap_diffs[lower_idx]
    ci_upper = bootstrap_diffs[upper_idx]

    # Compute p-value (two-tailed)
    # Under null hypothesis, diff should be centered at 0
    null_diffs = [d - observed_diff for d in bootstrap_diffs]
    extreme_count = sum(1 for d in null_diffs if abs(d) >= abs(observed_diff))
    p_value = extreme_count / n_bootstrap

    return BootstrapResult(
        mean_diff=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence,
    )


def compare_strategies(
    strategy_results: dict[str, Sequence[float]],
    baseline: str = "random",
    confidence: float = 0.95,
) -> dict[str, StatisticalTestResult]:
    """Compare multiple strategies against a baseline.

    Args:
        strategy_results: Dict mapping strategy names to AP values.
        baseline: Name of the baseline strategy.
        confidence: Confidence level for significance.

    Returns:
        Dictionary mapping strategy names to comparison results.
    """
    if baseline not in strategy_results:
        raise ValueError(f"Baseline strategy '{baseline}' not found")

    baseline_values = strategy_results[baseline]
    comparisons = {}

    for name, values in strategy_results.items():
        if name == baseline:
            continue

        # Use paired t-test if same length, otherwise bootstrap
        if len(values) == len(baseline_values):
            result = compare_models_paired_ttest(values, baseline_values, confidence)
        else:
            bootstrap = compare_models_bootstrap(values, baseline_values, confidence=confidence)
            # Convert bootstrap result to StatisticalTestResult format
            result = StatisticalTestResult(
                test_name="bootstrap",
                statistic=bootstrap.mean_diff,
                p_value=bootstrap.p_value,
                effect_size=compute_cohens_d(list(values), list(baseline_values)),
                is_significant=bootstrap.p_value < (1 - confidence),
                confidence_level=confidence,
                mean_diff=bootstrap.mean_diff,
                ci_lower=bootstrap.ci_lower,
                ci_upper=bootstrap.ci_upper,
                sample_size_a=len(values),
                sample_size_b=len(baseline_values),
            )

        comparisons[name] = result

    return comparisons


def analyze_permutation_sensitivity(
    permutation_aps: Sequence[tuple[int, float]],
) -> dict:
    """Analyze sensitivity to ICL example ordering.

    Args:
        permutation_aps: List of (permutation_id, AP) tuples.

    Returns:
        Dictionary with sensitivity analysis results.
    """
    import statistics

    if not permutation_aps:
        return {"error": "No permutation results"}

    aps = [ap for _, ap in permutation_aps]
    n = len(aps)

    mean_ap = statistics.mean(aps)
    std_ap = statistics.stdev(aps) if n > 1 else 0.0
    cv = std_ap / mean_ap if mean_ap > 0 else 0.0

    best_perm = max(permutation_aps, key=lambda x: x[1])
    worst_perm = min(permutation_aps, key=lambda x: x[1])

    return {
        "n_permutations": n,
        "mean_ap": mean_ap,
        "std_ap": std_ap,
        "coefficient_of_variation": cv,
        "min_ap": min(aps),
        "max_ap": max(aps),
        "range_ap": max(aps) - min(aps),
        "best_permutation_id": best_perm[0],
        "best_ap": best_perm[1],
        "worst_permutation_id": worst_perm[0],
        "worst_ap": worst_perm[1],
        "sensitivity_level": (
            "high" if cv > 0.1 else "moderate" if cv > 0.05 else "low"
        ),
    }
