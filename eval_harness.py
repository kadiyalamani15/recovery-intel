"""
eval_harness.py — 20-case evaluation harness for Recovery Intelligence Agent

Samples 20 dates from real health data covering 4 edge case categories:
  1. high_strain_low_hrv   — post-workout day with HRV well below baseline
  2. great_sleep_low_recovery — good sleep efficiency but low HRV/high RHR
  3. travel_disruption     — fragmented sleep, irregular timing
  4. sleep_debt            — consecutive nights of short sleep
  + normal days as baseline

Usage:
  python eval_harness.py [--run] [--summary]
  --run     Execute the pipeline on all 20 cases and log to Langfuse
  --summary Print stored results without re-running
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RESULTS_PATH = DATA_DIR / "eval_results.json"
COMPARISON_PATH = DATA_DIR / "eval_comparison.json"


# ── Case generation ────────────────────────────────────────────────────────────

def _load_signals() -> dict[str, Optional[pd.DataFrame]]:
    def _load(name):
        p = DATA_DIR / f"{name}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if "start" in df.columns:
            df["start"] = pd.to_datetime(df["start"], utc=True)
        if "end" in df.columns:
            df["end"] = pd.to_datetime(df["end"], utc=True)
        return df

    return {
        "hrv": _load("hrv"),
        "rhr": _load("resting_hr"),
        "sleep": _load("sleep"),
        "workouts": _load("workouts"),
    }


def _daily_hrv(hrv_df: pd.DataFrame) -> pd.Series:
    """Daily mean HRV keyed by date string."""
    hrv_df = hrv_df.copy()
    hrv_df["date"] = hrv_df["start"].dt.date
    return hrv_df.groupby("date")["value"].mean()


def _daily_rhr(rhr_df: pd.DataFrame) -> pd.Series:
    rhr_df = rhr_df.copy()
    rhr_df["date"] = rhr_df["start"].dt.date
    return rhr_df.groupby("date")["value"].mean()


def _daily_sleep_efficiency(sleep_df: pd.DataFrame) -> pd.Series:
    """Daily sleep efficiency % keyed by date of session start."""
    sleep_df = sleep_df.copy()
    sleep_df["date"] = sleep_df["start"].dt.date
    asleep_stages = ["core", "deep", "rem", "unspecified"]

    results = {}
    for d, grp in sleep_df.groupby("date"):
        asleep_min = grp[grp["stage"].isin(asleep_stages)]["duration_min"].sum()
        inbed_min = grp[grp["stage"] == "inbed"]["duration_min"].sum()
        if inbed_min == 0:
            inbed_min = asleep_min
        if inbed_min > 0:
            results[d] = (asleep_min / inbed_min) * 100
    return pd.Series(results)


def _daily_sleep_duration(sleep_df: pd.DataFrame) -> pd.Series:
    """Daily total sleep hours keyed by date."""
    sleep_df = sleep_df.copy()
    sleep_df["date"] = sleep_df["start"].dt.date
    asleep_stages = ["core", "deep", "rem", "unspecified"]
    asleep = sleep_df[sleep_df["stage"].isin(asleep_stages)]
    return asleep.groupby("date")["duration_min"].sum() / 60.0


def _daily_workout_energy(workout_df: pd.DataFrame) -> pd.Series:
    workout_df = workout_df.copy()
    workout_df["date"] = workout_df["start"].dt.date
    return workout_df.groupby("date")["energy_kcal"].sum()


def select_test_cases(signals: dict) -> list[dict]:
    """
    Select 20 dates from real data representing 4 edge categories + normal days.
    Returns list of {date, category, description} dicts.
    """
    hrv_df = signals["hrv"]
    rhr_df = signals["rhr"]
    sleep_df = signals["sleep"]
    workout_df = signals["workouts"]

    if hrv_df is None or rhr_df is None:
        raise RuntimeError("Missing hrv.parquet or resting_hr.parquet. Run parse_health.py first.")

    daily_hrv = _daily_hrv(hrv_df)
    daily_rhr = _daily_rhr(rhr_df)
    daily_eff = _daily_sleep_efficiency(sleep_df) if sleep_df is not None else pd.Series(dtype=float)
    daily_dur = _daily_sleep_duration(sleep_df) if sleep_df is not None else pd.Series(dtype=float)
    daily_energy = _daily_workout_energy(workout_df) if workout_df is not None else pd.Series(dtype=float)

    # Base pool: any day with both HRV and RHR (255 dates)
    base_dates = sorted(set(daily_hrv.index) & set(daily_rhr.index))
    # Sleep-enriched pool: days that also have tracked sleep (54 dates)
    sleep_dates = sorted(set(base_dates) & set(daily_eff.index))

    if len(base_dates) < 20:
        raise RuntimeError(f"Too few days with HRV+RHR ({len(base_dates)}). Run parse_health.py --days 730.")

    # Compute baselines (rolling 30-day, use overall for simplicity)
    hrv_mean = daily_hrv.mean()
    hrv_std = daily_hrv.std()
    rhr_mean = daily_rhr.mean()
    rhr_std = daily_rhr.std()

    cases = []
    used_dates = set()

    def _add(category: str, description: str, dates: list, max_n: int):
        added = 0
        for d in dates:
            ds = str(d)
            if ds not in used_dates and added < max_n:
                cases.append({"date": ds, "category": category, "description": description})
                used_dates.add(ds)
                added += 1
        return added

    # Category 1: High strain + Low HRV — any date with HRV well below baseline
    high_strain_candidates = sorted(
        [d for d in base_dates if daily_hrv.get(d, hrv_mean) < hrv_mean - 0.8 * hrv_std],
        key=lambda d: daily_hrv.get(d, hrv_mean),
    )
    _add("high_strain_low_hrv", "Post-workout HRV suppression", high_strain_candidates, 5)

    # Category 2: Great sleep + suppressed HRV — requires sleep-enriched pool
    great_sleep_candidates = [
        d for d in sleep_dates
        if daily_eff.get(d, 0) > 85 and daily_hrv.get(d, hrv_mean) < hrv_mean
    ]
    _add("great_sleep_low_recovery", "Good sleep, suppressed HRV", great_sleep_candidates, 5)

    # Category 3: Travel / disruption — very short or inefficient sleep
    disrupted_candidates = sorted(
        [d for d in sleep_dates if daily_dur.get(d, 7.0) < 5.5 or daily_eff.get(d, 80) < 72],
        key=lambda d: daily_eff.get(d, 80),
    )
    _add("travel_disruption", "Fragmented or short sleep", disrupted_candidates, 5)

    # Category 4: Sleep debt — 3+ consecutive nights short sleep in sleep_dates
    debt_candidates = []
    date_list = sorted(sleep_dates)
    for i in range(2, len(date_list)):
        d0, d1, d2 = date_list[i - 2], date_list[i - 1], date_list[i]
        consec = (
            (d1 - d0).days == 1 and (d2 - d1).days == 1
            and daily_dur.get(d0, 8) < 6.5
            and daily_dur.get(d1, 8) < 6.5
            and daily_dur.get(d2, 8) < 6.5
        )
        if consec:
            debt_candidates.append(d2)
    _add("sleep_debt", "3+ consecutive short sleep nights", debt_candidates, 3)

    # Fill remaining with normal days: near-baseline HRV from any HRV+RHR date
    needed = 20 - len(cases)
    normal_candidates = [
        d for d in base_dates
        if abs(daily_hrv.get(d, hrv_mean) - hrv_mean) < 0.4 * hrv_std
    ]
    step = max(1, len(normal_candidates) // (needed + 1))
    _add("normal", "Near-baseline day", normal_candidates[::step], needed)

    print(f"Selected {len(cases)} test cases:")
    for cat in ["normal", "high_strain_low_hrv", "great_sleep_low_recovery", "travel_disruption", "sleep_debt"]:
        n = sum(1 for c in cases if c["category"] == cat)
        print(f"  {cat}: {n} cases")

    return cases[:20]


# ── Run evaluation ─────────────────────────────────────────────────────────────

def run_eval(cases: list[dict]) -> list[dict]:
    from pipeline import run_pipeline

    lf = None
    try:
        from langfuse import Langfuse
        lf = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        pass

    results = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['date']} — {case['category']}: {case['description']}")
        try:
            result = run_pipeline(run_date=case["date"], eval_mode=True)
            eval_r = result.get("eval_results", {})
            pass_rate = result.get("eval_pass_rate", 0.0)
            score = result.get("recovery_score")

            # Log to Langfuse
            if lf:
                trace = lf.trace(
                    name="eval_harness",
                    tags=[case["category"], case["date"]],
                    input={"date": case["date"], "category": case["category"]},
                    output={
                        "recovery_score": score,
                        "eval_results": eval_r,
                        "pass_rate": pass_rate,
                    },
                )
                lf.score(trace_id=trace.id, name="pass_rate", value=pass_rate)
                for crit, r in eval_r.items():
                    lf.score(trace_id=trace.id, name=f"criterion_{crit}", value=r.get("score", 0))

            record = {
                **case,
                "recovery_score": score,
                "hrv_today": result.get("hrv_today"),
                "rhr_today": result.get("rhr_today"),
                "sleep_duration_h": result.get("sleep_duration_h"),
                "sleep_efficiency_pct": result.get("sleep_efficiency_pct"),
                "eval_pass_rate": pass_rate,
                "eval_results": eval_r,
                "insights": result.get("insights", []),
                "sleep_recommendation": result.get("sleep_recommendation"),
            }
            results.append(record)
            print(f"  Score: {score}/100 | Pass rate: {pass_rate:.0%}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({**case, "error": str(e), "eval_pass_rate": 0.0})

    if lf:
        lf.flush()

    return results


def save_results(results: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


def load_results() -> Optional[list[dict]]:
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


def load_comparison() -> Optional[dict]:
    if not COMPARISON_PATH.exists():
        return None
    with open(COMPARISON_PATH) as f:
        return json.load(f)


# ── Comparison eval (generic vs personalized) ──────────────────────────────────

def run_comparison_eval(cases: list[dict]) -> dict:
    """
    Run all 20 cases with both prompt variants.
    Returns aggregated comparison: per-criterion pass rates + delta.
    """
    from pipeline import run_pipeline, EVAL_CRITERIA

    case_results = []

    for i, case in enumerate(cases, 1):
        if i > 1:
            # Give the 6K TPM budget a moment to recover between cases
            time.sleep(30)
        print(f"\n[{i}/{len(cases)}] {case['date']} — {case['category']}")
        row = {**case, "generic": {}, "personalized": {}}

        for j, variant in enumerate(("generic", "personalized")):
            if j > 0:
                # Let default-model budget recover between variants within the same case
                time.sleep(12)
            print(f"  Running {variant}...", end=" ", flush=True)
            try:
                result = run_pipeline(
                    run_date=case["date"],
                    eval_mode=True,
                    prompt_variant=variant,
                )
                eval_r = result.get("eval_results", {})
                row[variant] = {
                    "eval_results": eval_r,
                    "eval_pass_rate": result.get("eval_pass_rate", 0.0),
                    "insights": result.get("insights", []),
                    "sleep_recommendation": result.get("sleep_recommendation", ""),
                    "recovery_score": result.get("recovery_score"),
                }
                print(f"{result.get('eval_pass_rate', 0):.0%}")
            except Exception as e:
                print(f"ERROR: {e}")
                row[variant] = {"error": str(e), "eval_pass_rate": 0.0, "eval_results": {}}

        case_results.append(row)

    # Aggregate pass rates per criterion per variant
    agg = {variant: {} for variant in ("generic", "personalized")}
    for variant in ("generic", "personalized"):
        from pipeline import EVAL_CRITERIA as _CRIT
        for crit in _CRIT:
            scores = [
                r[variant].get("eval_results", {}).get(crit, {}).get("score", 0)
                for r in case_results if "error" not in r[variant]
            ]
            agg[variant][crit] = round(sum(scores) / len(scores), 3) if scores else 0.0
        rates = [r[variant].get("eval_pass_rate", 0) for r in case_results if "error" not in r[variant]]
        agg[variant]["overall"] = round(sum(rates) / len(rates), 3) if rates else 0.0

    # Per-criterion delta
    from pipeline import EVAL_CRITERIA as _CRIT
    delta = {
        crit: round(agg["personalized"][crit] - agg["generic"][crit], 3)
        for crit in _CRIT
    }
    delta["overall"] = round(agg["personalized"]["overall"] - agg["generic"]["overall"], 3)

    comparison = {
        "cases": case_results,
        "aggregate": agg,
        "delta": delta,
        "n_cases": len(case_results),
    }

    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison saved to {COMPARISON_PATH}")

    return comparison


def print_comparison(comparison: dict) -> None:
    from pipeline import EVAL_CRITERIA
    agg = comparison.get("aggregate", {})
    delta = comparison.get("delta", {})
    n = comparison.get("n_cases", 0)

    labels = {
        "data_grounded": "Data Grounded",
        "actionable": "Actionable",
        "personally_calibrated": "Personally Calibrated",
        "science_aligned": "Science Aligned",
        "appropriately_confident": "Right Confidence",
        "overall": "OVERALL",
    }

    print("\n" + "=" * 68)
    print("PROMPT COMPARISON: Generic vs Personalized")
    print(f"{'':28} {'Generic':>8} {'Personal':>9} {'Delta':>7}")
    print("=" * 68)
    for crit in EVAL_CRITERIA + ["overall"]:
        g = agg.get("generic", {}).get(crit, 0)
        p = agg.get("personalized", {}).get(crit, 0)
        d = delta.get(crit, 0)
        arrow = "+" if d > 0 else ""
        label = labels.get(crit, crit)
        highlight = " <--" if crit == "personally_calibrated" else ""
        print(f"  {label:<26} {g:>7.0%} {p:>8.0%} {arrow}{d:>+6.0%}{highlight}")
    print("=" * 68)
    print(f"  n={n} cases across 4 edge categories")


def print_summary(results: list[dict]) -> None:
    from pipeline import EVAL_CRITERIA

    print("\n" + "=" * 60)
    print("EVAL HARNESS SUMMARY")
    print("=" * 60)

    rates = [r.get("eval_pass_rate", 0) for r in results if "error" not in r]
    overall = sum(rates) / len(rates) if rates else 0
    print(f"\nOverall pass rate: {overall:.0%}  ({sum(1 for r in rates if r == 1.0)}/{len(rates)} perfect)")

    # Per-criterion breakdown
    print("\nPer-criterion pass rate:")
    for crit in EVAL_CRITERIA:
        scores = [
            r.get("eval_results", {}).get(crit, {}).get("score", 0)
            for r in results if "error" not in r and r.get("eval_results")
        ]
        if scores:
            pct = sum(scores) / len(scores)
            bar = "█" * round(pct * 10) + "░" * (10 - round(pct * 10))
            print(f"  {crit:<28} {bar} {pct:.0%}")

    # By category
    print("\nBy category:")
    cats = {}
    for r in results:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, []).append(r.get("eval_pass_rate", 0))
    for cat, rates_ in cats.items():
        avg = sum(rates_) / len(rates_)
        print(f"  {cat:<30} {avg:.0%}  (n={len(rates_)})")

    print("\nCase details:")
    print(f"  {'Date':<12} {'Category':<28} {'Score':>5} {'Pass':>5}")
    print(f"  {'-'*12} {'-'*28} {'-'*5} {'-'*5}")
    for r in results:
        score = r.get("recovery_score", "ERR")
        rate = r.get("eval_pass_rate", 0)
        err = " ERROR" if "error" in r else ""
        print(f"  {r['date']:<12} {r['category']:<28} {str(score):>5} {rate:.0%}{err}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Run personalized-only eval")
    ap.add_argument("--compare", action="store_true", help="Run generic vs personalized comparison")
    ap.add_argument("--summary", action="store_true", help="Print stored results")
    ap.add_argument("--n", type=int, default=None, help="Limit to first N cases (default: all 20)")
    args = ap.parse_args()

    if args.summary:
        comp = load_comparison()
        if comp:
            print_comparison(comp)
        results = load_results()
        if results:
            print_summary(results)
        if not comp and not results:
            print("No results found. Run with --run or --compare first.")
        return

    if args.compare:
        signals = _load_signals()
        cases = select_test_cases(signals)
        if args.n:
            cases = cases[:args.n]
            print(f"  (limiting to first {args.n} cases)")
        comparison = run_comparison_eval(cases)
        print_comparison(comparison)
        return

    if args.run:
        signals = _load_signals()
        cases = select_test_cases(signals)
        if args.n:
            cases = cases[:args.n]
            print(f"  (limiting to first {args.n} cases)")
        results = run_eval(cases)
        save_results(results)
        print_summary(results)
    else:
        # Default: show cases without running
        signals = _load_signals()
        cases = select_test_cases(signals)
        print("\n20 test cases selected (use --run to execute):")
        for c in cases:
            print(f"  {c['date']}  {c['category']:<30} {c['description']}")


if __name__ == "__main__":
    main()
