"""
parse_health.py — Stream Apple Health export.xml → parquet files in data/

Records extracted (last 90 days):
  hrv.parquet          HKQuantityTypeIdentifierHeartRateVariabilitySDNN
  resting_hr.parquet   HKQuantityTypeIdentifierRestingHeartRate
  sleep.parquet        HKCategoryTypeIdentifierSleepAnalysis
  resp_rate.parquet    HKQuantityTypeIdentifierRespiratoryRate
  spo2.parquet         HKQuantityTypeIdentifierOxygenSaturation
  vo2max.parquet       HKQuantityTypeIdentifierVO2Max
  active_energy.parquet HKQuantityTypeIdentifierActiveEnergyBurned
  workouts.parquet     Workout elements

Usage:
  python parse_health.py [--xml PATH] [--days 90]
"""

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from lxml import etree

DATA_DIR = Path(__file__).parent / "data"

RECORD_TYPES = {
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
    "HKQuantityTypeIdentifierRespiratoryRate": "resp_rate",
    "HKQuantityTypeIdentifierOxygenSaturation": "spo2",
    "HKQuantityTypeIdentifierVO2Max": "vo2max",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy",
}

SLEEP_STAGES = {
    "HKCategoryValueSleepAnalysisAsleepCore": "core",
    "HKCategoryValueSleepAnalysisAsleepDeep": "deep",
    "HKCategoryValueSleepAnalysisAsleepREM": "rem",
    "HKCategoryValueSleepAnalysisAsleepUnspecified": "unspecified",
    "HKCategoryValueSleepAnalysisAwake": "awake",
    "HKCategoryValueSleepAnalysisInBed": "inbed",
}


def parse_date(s: str) -> datetime:
    """Parse Apple Health date strings like '2024-01-16 19:25:19 -0400'."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")


def stream_parse(xml_path: str, cutoff_dt: datetime) -> dict[str, list]:
    """
    Stream through export.xml with iterparse.
    Returns dict of signal_name → list of row dicts.
    Memory-efficient: processes one element at a time, discards after use.
    """
    buckets: dict[str, list] = {name: [] for name in RECORD_TYPES.values()}
    buckets["workouts"] = []

    total_records = 0
    kept_records = 0

    print(f"Parsing {xml_path} (cutoff: {cutoff_dt.date()})...")

    context = etree.iterparse(xml_path, events=("end",), tag=("Record", "Workout"))

    for _event, elem in context:
        total_records += 1
        if total_records % 100_000 == 0:
            print(f"  Processed {total_records:,} elements, kept {kept_records:,}...", flush=True)

        if elem.tag == "Record":
            rec_type = elem.get("type", "")
            signal = RECORD_TYPES.get(rec_type)
            if signal is None:
                elem.clear()
                continue

            start_str = elem.get("startDate", "")
            try:
                start_dt = parse_date(start_str)
            except ValueError:
                elem.clear()
                continue

            if start_dt < cutoff_dt:
                elem.clear()
                continue

            end_str = elem.get("endDate", "")
            try:
                end_dt = parse_date(end_str)
            except ValueError:
                end_dt = start_dt

            row = {
                "start": start_dt,
                "end": end_dt,
                "value": elem.get("value"),
                "unit": elem.get("unit", ""),
                "source": elem.get("sourceName", ""),
            }

            # Sleep: decode value to stage name
            if signal == "sleep":
                row["stage"] = SLEEP_STAGES.get(row["value"], "unknown")
                row["duration_min"] = (end_dt - start_dt).total_seconds() / 60.0

            # Numeric conversion for quantity types
            if signal != "sleep" and row["value"] is not None:
                try:
                    row["value"] = float(row["value"])
                except ValueError:
                    pass

            buckets[signal].append(row)
            kept_records += 1

        elif elem.tag == "Workout":
            start_str = elem.get("startDate", "")
            try:
                start_dt = parse_date(start_str)
            except ValueError:
                elem.clear()
                continue

            if start_dt < cutoff_dt:
                elem.clear()
                continue

            end_str = elem.get("endDate", "")
            try:
                end_dt = parse_date(end_str)
            except ValueError:
                end_dt = start_dt

            duration_raw = elem.get("duration")
            energy_raw = elem.get("totalEnergyBurned")

            row = {
                "start": start_dt,
                "end": end_dt,
                "type": elem.get("workoutActivityType", ""),
                "duration_min": float(duration_raw) if duration_raw else None,
                "energy_kcal": float(energy_raw) if energy_raw else None,
                "source": elem.get("sourceName", ""),
            }
            buckets["workouts"].append(row)
            kept_records += 1

        elem.clear()

    print(f"Done. Processed {total_records:,} elements, kept {kept_records:,}.")
    return buckets


def save_parquets(buckets: dict[str, list]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, rows in buckets.items():
        if not rows:
            print(f"  {name}: 0 records — skipping")
            continue
        df = pd.DataFrame(rows)
        # Normalize timezone-aware datetimes to UTC for parquet compatibility
        for col in ("start", "end"):
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_convert("UTC")
        path = DATA_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {name}: {len(df):,} records → {path}")


def main():
    parser = argparse.ArgumentParser(description="Parse Apple Health XML export")
    parser.add_argument(
        "--xml",
        default=os.getenv("HEALTH_EXPORT_PATH", "/tmp/apple-health-explore/apple_health_export/export.xml"),
        help="Path to export.xml",
    )
    parser.add_argument("--days", type=int, default=365, help="Days of history to keep (use 730 for full 2-year baseline)")
    args = parser.parse_args()

    if not Path(args.xml).exists():
        print(f"ERROR: {args.xml} not found.")
        sys.exit(1)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    buckets = stream_parse(args.xml, cutoff)
    save_parquets(buckets)
    print("\nDone. Run: python pipeline.py  or  streamlit run app.py")


if __name__ == "__main__":
    main()
