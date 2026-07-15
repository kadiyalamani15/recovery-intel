# Recovery Intel

Personal recovery intelligence agent built on Apple Health data + Claude + LangGraph.

Built as a portfolio project for WHOOP Senior PM, Sleep (report 017).

## Quick Start

```bash
# 1. Clone & install
cd recovery-intel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and Langfuse keys

# 3. Parse Apple Health export
python parse_health.py --xml /path/to/export.xml --days 730

# 4. Run the Streamlit app
streamlit run app.py

# 5. Run eval harness (requires API key, takes ~5-10 min)
python eval_harness.py --run
python eval_harness.py --summary
```

## Architecture

```
Apple Health export.xml  →  parse_health.py  →  data/*.parquet
                                                       ↓
                                              pipeline.py (LangGraph 5 nodes)
                                                       ↓
                                              app.py (Streamlit 3 tabs)
                                              eval_harness.py (20 test cases)
```

### LangGraph Pipeline (5 nodes)

| Node | Input | Output |
|------|-------|--------|
| `build_context` | date | HRV/RHR vs 30-day baseline, last night sleep, 7-day workout load |
| `score_recovery` | context | Recovery score 0-100 (HRV 40% + RHR 30% + sleep efficiency 30%) |
| `generate_insights` | context + score | 2-3 data-grounded observations |
| `coach_sleep` | context + score | Tonight's sleep target + bedtime |
| `evaluate_quality` | all outputs | LLM-as-judge on 5 criteria (eval mode only) |

### Recovery Score Formula

```
hrv_score  = clamp(50 + 25 * z_hrv, 0, 100)   # higher HRV = better
rhr_score  = clamp(50 + 25 * z_rhr_inv, 0, 100) # lower RHR = better
sleep_score = clamp((efficiency% - 60) / 30 * 100, 0, 100)

recovery_score = 0.40 * hrv_score + 0.30 * rhr_score + 0.30 * sleep_score
```

### Eval Harness

20 test cases from real health history:
- **5 normal** — near-baseline HRV days
- **5 high_strain_low_hrv** — post-workout HRV suppression
- **5 great_sleep_low_recovery** — good sleep efficiency, suppressed HRV
- **5 travel_disruption** — fragmented or short sleep sessions

LLM-as-judge criteria:
1. **data_grounded** — cites actual numbers (HRV ms, HR bpm, sleep hours)
2. **actionable** — gives specific, doable guidance
3. **personally_calibrated** — references user's own baseline, not population norms
4. **science_aligned** — consistent with sleep/recovery science
5. **appropriately_confident** — avoids false certainty and excessive hedging

All eval traces logged to Langfuse for the dashboard.

## Signals Used

| Signal | Records | Source |
|--------|---------|--------|
| HRV SDNN | ~1,200 | Apple Watch |
| Resting Heart Rate | ~270 | Apple Watch |
| Sleep Analysis | ~1,440 | Apple Watch |
| Respiratory Rate | ~2,000 | Apple Watch |
| SpO2 | ~1,000 | Apple Watch |
| VO2 Max | ~50 | Apple Watch |
| Active Energy | ~60,000 | iPhone + Watch |
| Workouts | ~110 | Apple Watch |

## Daily Refresh

For daily fresh data without re-exporting the full XML:
1. Use **Health Auto Export** ($4 app) to auto-export JSON to iCloud Drive
2. Or set up an Apple Shortcut at 7am to write key metrics to a JSON file

(Phase 2 automation — the insight layer works independently of this.)
