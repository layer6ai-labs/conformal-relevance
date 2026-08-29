"""Extract experiment data from DB for paper appendices, tables, and bootstrap CIs.

Usage:
    uv run python -m conformal_relevance.evaluation.paper_data
    uv run python -m conformal_relevance.evaluation.paper_data --output path/to/output.json

Output JSON sections:
    strategy_matrix:  latest MAP per (dataset, strategy) at default config
    ap_distributions: per-sample AP summary stats for ICL0 vs Ens4
    bootstrap_cis:    95% bootstrap CIs on MAP for key comparisons
    reproducibility:  run IDs, latency, sample counts for Appendix H
"""

from __future__ import annotations

import json
import statistics as pystats
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from conformal_relevance.db import get_session
from conformal_relevance.db.operations import get_run_samples
from conformal_relevance.evaluation.queries import (
    get_performance_matrix,
    get_run_summary,
)
from conformal_relevance.evaluation.statistics import (
    bootstrap_map_ci,
    compare_models_bootstrap,
)


DATASETS = [
    "ectsum", "subsume", "hotpotqa", "physionet", "puma",
    "evidence_inference", "contractnli",
]

DISPLAY_NAMES = {
    "ectsum": "ECTSum",
    "subsume": "SubSumE",
    "hotpotqa": "HotpotQA",
    "physionet": "PhysioNet",
    "puma": "PUMA",
    "evidence_inference": "Evidence Inf.",
    "contractnli": "ContractNLI",
}

# Per-dataset Best Single configuration (post-hoc oracle: max MAP over
# {anchor_dpp, bm25, pattern_dpp, random} x k in {1,2,3,5,8}). Keys match
# strategy_matrix labels (icl{k}_{strategy}).
BEST_SINGLE_CONFIGS = {
    "ectsum": "icl3_anchor_dpp",
    "subsume": "icl3_bm25",
    "hotpotqa": "icl2_anchor_dpp",
    "physionet": "icl3_random",
    "puma": "icl2_anchor_dpp",
    "evidence_inference": "icl1_anchor_dpp",
    "contractnli": "icl3_pattern_dpp",
}


def compute_ap_distribution_stats(ap_values: Sequence[float]) -> dict[str, Any]:
    """Summary statistics for a per-sample AP distribution.

    Returns dict with n, mean, std, min, p5, p25, p50, p75, p95, max,
    pct_zero (% of samples with AP=0), pct_perfect (% with AP=1).
    """
    values = sorted(ap_values)
    n = len(values)
    if n == 0:
        return {"n": 0}

    mean = pystats.mean(values)
    std = pystats.pstdev(values)

    def percentile(p: float) -> float:
        k = (n - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < n else f
        d = k - f
        return values[f] + d * (values[c] - values[f])

    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(values[0], 4),
        "p5": round(percentile(5), 4),
        "p25": round(percentile(25), 4),
        "p50": round(percentile(50), 4),
        "p75": round(percentile(75), 4),
        "p95": round(percentile(95), 4),
        "max": round(values[-1], 4),
        "pct_zero": round(sum(1 for v in values if v == 0) / n * 100, 1),
        "pct_perfect": round(sum(1 for v in values if v >= 1.0) / n * 100, 1),
    }


def extract_paper_data(
    model_name: str = "gemini-2.5-flash-lite",
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Extract all paper data from the database."""
    output: dict[str, Any] = {
        "model": model_name,
        "n_bootstrap": n_bootstrap,
        "strategy_matrix": {},
        "ap_distributions": {},
        "bootstrap_cis": {},
        "reproducibility": {},
    }

    with get_session() as session:
        # 1. Strategy x dataset performance matrix (latest runs)
        perf = get_performance_matrix(session, model_name=model_name)
        by_dataset: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in perf:
            ds = row["dataset_name"]
            label = row["strategy_label"]
            by_dataset[ds][label] = {
                "mean_ap": row["mean_ap"],
                "std_ap": row["std_ap"],
                "run_id": row["run_id"],
                "n_samples": row["n_samples"],
            }
        output["strategy_matrix"] = dict(by_dataset)

        # 2. Per-sample AP distributions + bootstrap CIs for ICL0 vs Ens4
        for ds in DATASETS:
            ds_data = by_dataset.get(ds, {})

            icl0_info = ds_data.get("icl0")
            ens4_info = ds_data.get("icl2_moe4") or ds_data.get("icl2_moe")

            if not icl0_info or not ens4_info:
                print(f"WARNING: Missing ICL0 or Ens4 run for {ds}")
                print(f"  Available: {list(ds_data.keys())}")
                continue

            icl0_samples = get_run_samples(session, icl0_info["run_id"], status="scored")
            ens4_samples = get_run_samples(session, ens4_info["run_id"], status="scored")

            icl0_aps = [s.ap for s in icl0_samples if s.ap is not None]
            ens4_aps = [s.ap for s in ens4_samples if s.ap is not None]

            if not icl0_aps or not ens4_aps:
                print(f"WARNING: No per-sample APs for {ds}")
                continue

            # AP distribution stats
            output["ap_distributions"][ds] = {
                "icl0": compute_ap_distribution_stats(icl0_aps),
                "ens4": compute_ap_distribution_stats(ens4_aps),
                "icl0_run_id": icl0_info["run_id"],
                "ens4_run_id": ens4_info["run_id"],
            }

            # Bootstrap CIs - single-distribution (MAP uncertainty)
            icl0_ci = bootstrap_map_ci(icl0_aps, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
            ens4_ci = bootstrap_map_ci(ens4_aps, n_bootstrap=n_bootstrap, seed=bootstrap_seed)

            # Bootstrap CI - paired difference (Ens4 - ICL0)
            icl0_by_idx = {s.sample_index: s.ap for s in icl0_samples if s.ap is not None}
            ens4_by_idx = {s.sample_index: s.ap for s in ens4_samples if s.ap is not None}
            common = sorted(set(icl0_by_idx) & set(ens4_by_idx))

            paired_ci = None
            if common:
                paired_result = compare_models_bootstrap(
                    [ens4_by_idx[i] for i in common],
                    [icl0_by_idx[i] for i in common],
                    n_bootstrap=n_bootstrap, seed=bootstrap_seed,
                )
                paired_ci = paired_result.to_dict()

            # Bootstrap CI - paired difference (Ens4 - Best Single)
            best_single_label = BEST_SINGLE_CONFIGS.get(ds)
            best_single_info = ds_data.get(best_single_label) if best_single_label else None
            best_single_diff = None
            best_single_run_id = None
            if best_single_info:
                best_single_run_id = best_single_info["run_id"]
                bs_samples = get_run_samples(session, best_single_run_id, status="scored")
                bs_by_idx = {s.sample_index: s.ap for s in bs_samples if s.ap is not None}
                bs_common = sorted(set(bs_by_idx) & set(ens4_by_idx))
                if bs_common:
                    bs_result = compare_models_bootstrap(
                        [ens4_by_idx[i] for i in bs_common],
                        [bs_by_idx[i] for i in bs_common],
                        n_bootstrap=n_bootstrap, seed=bootstrap_seed,
                    )
                    best_single_diff = bs_result.to_dict()
                    best_single_diff["best_single_label"] = best_single_label
                    best_single_diff["best_single_run_id"] = best_single_run_id
            else:
                print(f"WARNING: Best Single config not found for {ds} "
                      f"(expected label {best_single_label!r})")

            output["bootstrap_cis"][ds] = {
                "icl0": icl0_ci.to_dict(),
                "ens4": ens4_ci.to_dict(),
                "paired_diff": paired_ci,
                "paired_diff_best_single": best_single_diff,
            }

            # 3. Reproducibility metadata
            icl0_summary = get_run_summary(session, icl0_info["run_id"])
            ens4_summary = get_run_summary(session, ens4_info["run_id"])

            def _repro_entry(summary: dict | None) -> dict:
                if not summary:
                    return {}
                return {
                    "run_id": summary.get("run_id"),
                    "n_samples": summary.get("n_samples"),
                    "total_duration_ms": summary.get("total_duration_ms"),
                    "total_llm_time_ms": summary.get("total_llm_time_ms"),
                    "avg_sample_time_ms": summary.get("avg_sample_time_ms"),
                    "model": summary.get("model_name"),
                    "strategy": summary.get("icl_strategy_name"),
                }

            output["reproducibility"][ds] = {
                "icl0": _repro_entry(icl0_summary),
                "ens4": _repro_entry(ens4_summary),
            }

        # 4. Enrich strategy_matrix entries with run-level metadata
        for ds in DATASETS:
            ds_data = by_dataset.get(ds, {})
            for label, info in ds_data.items():
                run_summary = get_run_summary(session, info["run_id"])
                if run_summary:
                    info["ap_percentiles"] = run_summary.get("ap_percentiles")
                    info["total_duration_ms"] = run_summary.get("total_duration_ms")

    return output


def print_summary(data: dict[str, Any]) -> None:
    """Print a human-readable summary of extracted data."""
    print("\n=== Strategy Matrix ===")
    for ds in DATASETS:
        if ds in data["strategy_matrix"]:
            print(f"\n{DISPLAY_NAMES[ds]}:")
            for label, info in sorted(data["strategy_matrix"][ds].items()):
                map_val = info["mean_ap"]
                if map_val is not None:
                    print(f"  {label:25s} MAP={map_val:.3f}  (run_id={info['run_id']})")

    print("\n=== Bootstrap CIs (Ens4 vs ICL0) ===")
    for ds in DATASETS:
        if ds in data["bootstrap_cis"]:
            ci = data["bootstrap_cis"][ds]
            icl0 = ci["icl0"]
            ens4 = ci["ens4"]
            diff = ci.get("paired_diff") or {}
            print(
                f"{DISPLAY_NAMES[ds]:15s}  "
                f"ICL0={icl0.get('mean',0):.3f} [{icl0.get('ci_lower',0):.3f}, {icl0.get('ci_upper',0):.3f}]  "
                f"Ens4={ens4.get('mean',0):.3f} [{ens4.get('ci_lower',0):.3f}, {ens4.get('ci_upper',0):.3f}]  "
                f"Delta={diff.get('mean_diff',0):.3f} [{diff.get('ci_lower',0):.3f}, {diff.get('ci_upper',0):.3f}]"
            )

    print("\n=== Bootstrap CIs (Ens4 vs Best Single) ===")
    for ds in DATASETS:
        if ds in data["bootstrap_cis"]:
            diff = data["bootstrap_cis"][ds].get("paired_diff_best_single") or {}
            if diff:
                print(
                    f"{DISPLAY_NAMES[ds]:15s}  "
                    f"BestSingle={diff.get('best_single_label','?'):22s}  "
                    f"Delta={diff.get('mean_diff',0):+.3f} "
                    f"[{diff.get('ci_lower',0):+.3f}, {diff.get('ci_upper',0):+.3f}]"
                )

    print("\n=== AP Distributions (Ens4) ===")
    for ds in DATASETS:
        if ds in data["ap_distributions"]:
            d = data["ap_distributions"][ds]["ens4"]
            print(
                f"{DISPLAY_NAMES[ds]:15s}  "
                f"mean={d['mean']:.3f} std={d['std']:.3f}  "
                f"p5={d['p5']:.3f} p50={d['p50']:.3f} p95={d['p95']:.3f}  "
                f"zero%={d['pct_zero']:.1f}"
            )


def main() -> None:
    """CLI entry point: extract data and write to JSON."""
    import argparse

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Extract paper data from experiment DB")
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).resolve().parent.parent.parent / "paper" / "paper_data.json"),
        help="Output JSON path (default: paper/paper_data.json)",
    )
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    data = extract_paper_data(model_name=args.model, n_bootstrap=args.n_bootstrap)

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Wrote {outpath}")
    print_summary(data)


if __name__ == "__main__":
    main()
