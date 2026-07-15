"""
pipeline.py — Recovery Intelligence Agent (LangGraph 5-node pipeline)

Nodes:
  1. build_context    — last night sleep + HRV/RHR vs 30-day baseline + 7-day workout load
  2. score_recovery   — composite score 0-100 (HRV 40% + RHR 30% + sleep efficiency 30%)
  3. generate_insights — 2-3 Claude-generated observations grounded in actual numbers
  4. coach_sleep      — tonight's sleep target calibrated to personal baseline
  5. evaluate_quality — LLM-as-judge on 5 criteria (only when eval=True)

Usage:
  from pipeline import run_pipeline
  result = run_pipeline(date="2026-07-15")
"""

import os
import time
from datetime import datetime, timezone, timedelta, date as date_type
from pathlib import Path
from typing import TypedDict, Optional, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import anthropic
from langgraph.graph import StateGraph, END

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"

# ── Langfuse (optional) ────────────────────────────────────────────────────────

def _get_langfuse():
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    if not sk or not pk:
        return None
    try:
        from langfuse import Langfuse
        lf = Langfuse(
            secret_key=sk,
            public_key=pk,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        return lf
    except Exception:
        return None


_lf = None  # initialized lazily

def _langfuse():
    global _lf
    if _lf is None:
        _lf = _get_langfuse()
    return _lf


# ── State ─────────────────────────────────────────────────────────────────────

class RecoveryState(TypedDict, total=False):
    # Input
    run_date: str          # YYYY-MM-DD
    eval_mode: bool        # if True, node 5 runs
    prompt_variant: str    # "personalized" (default) | "generic"

    # Node 1: Context
    hrv_today: float       # morning average (ms)
    hrv_baseline: float    # 30-day mean (ms)
    hrv_std: float         # 30-day std (ms)
    hrv_pct_vs_baseline: float  # % deviation
    rhr_today: float       # today's resting HR (bpm)
    rhr_baseline: float    # 30-day mean
    rhr_std: float
    rhr_pct_vs_baseline: float
    sleep_duration_h: float       # last night total sleep hours
    sleep_inbed_h: float
    sleep_efficiency_pct: float   # asleep / inbed %
    sleep_stages: dict            # {core, deep, rem, awake} in minutes
    sleep_date: str               # date of last sleep session
    workout_load_7d_kcal: float   # 7-day active energy burned
    workout_count_7d: int

    # Node 2: Recovery Score
    recovery_score: int           # 0-100
    hrv_component: float          # 0-100
    rhr_component: float          # 0-100
    sleep_component: float        # 0-100
    recovery_label: str           # Optimal / Good / Moderate / Low

    # Node 3: Insights
    insights: list                # list of 2-3 strings

    # Node 4: Sleep Coach
    sleep_recommendation: str
    target_bedtime: str
    target_duration_h: float

    # Node 5: Eval
    eval_results: dict            # {criterion: {score: 0|1, reasoning: str}}
    eval_pass_rate: float


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load(name: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Ensure start is UTC-aware datetime
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"], utc=True)
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"], utc=True)
    return df


def _window(df: pd.DataFrame, days: int, ref_dt: datetime) -> pd.DataFrame:
    cutoff = ref_dt - timedelta(days=days)
    return df[df["start"] >= cutoff].copy()


def _morning_hrv(hrv_df: pd.DataFrame, target_date: str) -> Optional[float]:
    """Average HRV readings between midnight and noon on target_date (local proxy: UTC)."""
    dt = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    mask = (hrv_df["start"] >= dt) & (hrv_df["start"] < dt + timedelta(hours=14))
    readings = hrv_df[mask]["value"].dropna()
    return float(readings.mean()) if len(readings) > 0 else None


def _last_night_sleep(sleep_df: pd.DataFrame, ref_dt: datetime) -> dict:
    """
    Extract the most recent sleep session relative to ref_dt.
    Looks back up to 36 hours from ref_dt to find the latest session,
    regardless of whether it started before or after the ref_dt anchor.
    Groups records within 4-hour gaps into sessions.
    """
    # Look back 36h from end-of-next-day to catch tonight's session if in progress
    search_end = ref_dt + timedelta(hours=28)
    search_start = ref_dt - timedelta(hours=36)
    window = sleep_df[
        (sleep_df["start"] >= search_start) & (sleep_df["start"] <= search_end)
    ].copy()

    if window.empty:
        # Fallback: find the absolute most recent session in the entire dataset
        window = sleep_df.copy()
        if window.empty:
            return {}

    window = window.sort_values("start")
    # Find the most recent session: work backwards from the latest start
    latest_end = window["end"].max()
    # A "session" = records that start within 18h of latest_end
    session_cutoff = latest_end - timedelta(hours=18)
    session = window[window["start"] >= session_cutoff]

    asleep_stages = ["core", "deep", "rem", "unspecified"]
    asleep = session[session["stage"].isin(asleep_stages)]
    inbed = session[session["stage"] == "inbed"]

    # Total minutes
    asleep_min = asleep["duration_min"].sum()
    inbed_min = inbed["duration_min"].sum() if not inbed.empty else asleep_min

    if inbed_min == 0:
        inbed_min = asleep_min

    stage_breakdown = {}
    for stage in ["core", "deep", "rem", "awake"]:
        stage_breakdown[stage] = round(session[session["stage"] == stage]["duration_min"].sum())

    sleep_date_str = session["start"].min().strftime("%Y-%m-%d") if not session.empty else ""

    return {
        "asleep_min": round(asleep_min),
        "inbed_min": round(inbed_min),
        "efficiency_pct": round((asleep_min / inbed_min) * 100, 1) if inbed_min > 0 else 0.0,
        "stages": stage_breakdown,
        "date": sleep_date_str,
    }


# ── Node 1: build_context ─────────────────────────────────────────────────────

def build_context(state: RecoveryState) -> RecoveryState:
    run_date = state.get("run_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ref_dt = datetime.strptime(run_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(hours=20)

    updates: RecoveryState = {}

    # HRV
    hrv_df = _load("hrv")
    if hrv_df is not None:
        baseline_df = _window(hrv_df, 30, ref_dt)
        hrv_today = _morning_hrv(hrv_df, run_date)
        hrv_mean = float(baseline_df["value"].mean()) if not baseline_df.empty else 60.0
        hrv_std = float(baseline_df["value"].std()) if len(baseline_df) > 1 else 10.0
        if hrv_today is None:
            # Fallback: use yesterday's last reading
            yesterday = (datetime.strptime(run_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            hrv_today = _morning_hrv(hrv_df, yesterday) or hrv_mean
        updates["hrv_today"] = round(hrv_today, 1)
        updates["hrv_baseline"] = round(hrv_mean, 1)
        updates["hrv_std"] = round(hrv_std, 1)
        updates["hrv_pct_vs_baseline"] = round((hrv_today - hrv_mean) / hrv_mean * 100, 1)

    # RHR
    rhr_df = _load("resting_hr")
    if rhr_df is not None:
        baseline_df = _window(rhr_df, 30, ref_dt)
        today_rhr_rows = rhr_df[
            (rhr_df["start"] >= ref_dt - timedelta(hours=36)) &
            (rhr_df["start"] < ref_dt)
        ]
        rhr_today = float(today_rhr_rows["value"].iloc[-1]) if not today_rhr_rows.empty else None
        rhr_mean = float(baseline_df["value"].mean()) if not baseline_df.empty else 55.0
        rhr_std = float(baseline_df["value"].std()) if len(baseline_df) > 1 else 5.0
        if rhr_today is None:
            rhr_today = rhr_mean
        updates["rhr_today"] = round(rhr_today, 1)
        updates["rhr_baseline"] = round(rhr_mean, 1)
        updates["rhr_std"] = round(rhr_std, 1)
        updates["rhr_pct_vs_baseline"] = round((rhr_mean - rhr_today) / rhr_mean * 100, 1)

    # Sleep
    sleep_df = _load("sleep")
    if sleep_df is not None:
        session = _last_night_sleep(sleep_df, ref_dt)
        if session:
            asleep_h = session["asleep_min"] / 60
            inbed_h = session["inbed_min"] / 60
            updates["sleep_duration_h"] = round(asleep_h, 2)
            updates["sleep_inbed_h"] = round(inbed_h, 2)
            updates["sleep_efficiency_pct"] = session["efficiency_pct"]
            updates["sleep_stages"] = session["stages"]
            updates["sleep_date"] = session["date"]

    # Workout load (7 days) — use active_energy for kcal (workouts often lack totalEnergyBurned)
    workout_df = _load("workouts")
    active_df = _load("active_energy")
    if workout_df is not None:
        w7 = _window(workout_df, 7, ref_dt)
        updates["workout_count_7d"] = int(len(w7))
    else:
        updates["workout_count_7d"] = 0
    if active_df is not None:
        a7 = _window(active_df, 7, ref_dt)
        updates["workout_load_7d_kcal"] = round(float(a7["value"].sum()), 0)
    else:
        updates["workout_load_7d_kcal"] = 0.0

    return {**state, **updates}


# ── Node 2: score_recovery ────────────────────────────────────────────────────

def _sigmoid_score(z: float) -> float:
    """Map z-score to 0-100 via sigmoid-like clamp."""
    # z=-2 → ~0, z=0 → 50, z=+2 → ~100
    raw = 50 + 25 * z
    return min(100.0, max(0.0, raw))


def score_recovery(state: RecoveryState) -> RecoveryState:
    # HRV component (40%) — higher HRV = better
    hrv_today = state.get("hrv_today", state.get("hrv_baseline", 60))
    hrv_baseline = state.get("hrv_baseline", 60)
    hrv_std = state.get("hrv_std", 10) or 10
    hrv_z = (hrv_today - hrv_baseline) / hrv_std
    hrv_component = _sigmoid_score(hrv_z)

    # RHR component (30%) — lower RHR = better (sign inverted)
    rhr_today = state.get("rhr_today", state.get("rhr_baseline", 55))
    rhr_baseline = state.get("rhr_baseline", 55)
    rhr_std = state.get("rhr_std", 5) or 5
    rhr_z = (rhr_baseline - rhr_today) / rhr_std  # positive = below baseline = good
    rhr_component = _sigmoid_score(rhr_z)

    # Sleep efficiency component (30%)
    eff = state.get("sleep_efficiency_pct", 80.0)
    # 90%+ = 100pts, 75% = 50pts, 60%- = 0pts
    sleep_component = min(100.0, max(0.0, (eff - 60) / (90 - 60) * 100))

    # Composite
    score = 0.40 * hrv_component + 0.30 * rhr_component + 0.30 * sleep_component
    score = round(score)

    label = (
        "Optimal" if score >= 80
        else "Good" if score >= 65
        else "Moderate" if score >= 45
        else "Low"
    )

    return {
        **state,
        "recovery_score": score,
        "hrv_component": round(hrv_component),
        "rhr_component": round(rhr_component),
        "sleep_component": round(sleep_component),
        "recovery_label": label,
    }


# ── LLM client — Groq preferred, Anthropic fallback ───────────────────────────

# Groq model mapping (fast + capable)
_GROQ_MODELS = {
    "default": "llama-3.1-8b-instant",   # insights + coaching (30K TPM free tier)
    "judge": "llama-3.1-8b-instant",     # eval judge (30K TPM)
    "fast": "llama-3.1-8b-instant",
}

_groq_client = None
_anthropic_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _llm(system: str, user: str, model: str = "default", max_tokens: int = 512) -> str:
    """
    Call LLM. Prefers Groq if GROQ_API_KEY is set, falls back to Anthropic.
    model: "default" | "judge" | a literal Groq/Claude model ID
    Retries up to 3 times on rate-limit errors with exponential backoff.
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if groq_key:
        groq_model = _GROQ_MODELS.get(model, model if model not in _GROQ_MODELS else _GROQ_MODELS["default"])
        client = _get_groq()
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "rate limit" in err_str or "429" in err_str:
                    wait = 15 * (2 ** attempt)  # 15s, 30s, 60s
                    print(f"[rate limit] sleeping {wait}s before retry {attempt + 1}/3 …")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Groq rate limit not resolved after 3 retries for model {groq_model}")

    if anthropic_key:
        claude_model = "claude-opus-4-6" if model in ("default", "judge") else model
        client = _get_anthropic()
        msg = client.messages.create(
            model=claude_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()

    raise RuntimeError("No LLM API key configured. Set GROQ_API_KEY or ANTHROPIC_API_KEY in .env")


def _parse_json(raw: str) -> any:
    """
    Extract JSON from an LLM response robustly.
    Handles: plain JSON, ```json blocks, leading prose, truncated arrays.
    """
    import json, re
    text = raw.strip()

    # Strip code fences
    if "```" in text:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            text = fenced.group(1).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first JSON object or array in the text
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

    # Truncated array recovery: extract complete quoted strings
    if text.lstrip().startswith("["):
        strings = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
        # Filter out short fragments that look like JSON keys
        items = [s for s in strings if len(s) > 20]
        if items:
            return items

    raise ValueError(f"No valid JSON found in response: {text[:200]!r}")


# ── Node 3: generate_insights ──────────────────────────────────────────────────

# Personalized: instructs to compare against personal baseline
_INSIGHTS_SYSTEM_PERSONALIZED = """You are a recovery analyst interpreting wearable health data.
Your job is to generate 2-3 specific, data-grounded observations about today's recovery state.

Rules:
- Reference actual numbers from the data (HRV in ms, HR in bpm, sleep hours/percentages)
- Compare to the user's personal baseline, not population averages
- No generic advice ("sleep 8 hours", "exercise regularly") — be specific to this person's data
- Each insight is 1-2 sentences
- If data is below baseline, explain the physiological implication concisely
- Tone: clear, confident, not alarming
- Return exactly a JSON array of strings: ["insight 1", "insight 2", "insight 3"]"""

# Generic: population norms language, no personal baseline context in prompt or user message
_INSIGHTS_SYSTEM_GENERIC = """You are a health and wellness AI assistant providing recovery insights based on biometric data.

Rules:
- Reference the user's data values in your insights
- Provide context using established health guidelines and clinical reference ranges
- HRV context: 20-50ms is low, 50-100ms is moderate, 100ms+ is high for adults
- Resting HR context: under 60 bpm is athletic/healthy, 60-100 is normal
- Sleep efficiency context: above 85% is healthy, below 75% may indicate disrupted sleep
- Each insight is 1-2 sentences
- Tone: helpful and informative
- Return exactly a JSON array of strings: ["insight 1", "insight 2", "insight 3"]"""


def _insights_user_msg(state: RecoveryState, variant: str) -> str:
    if variant == "generic":
        # Generic: raw numbers only, no personal baseline
        return f"""Today's biometric data ({state.get('run_date')}):

HRV: {state.get('hrv_today')} ms
Resting HR: {state.get('rhr_today')} bpm

Last night's sleep:
  Total sleep: {state.get('sleep_duration_h', 0):.1f} hours
  Sleep efficiency: {state.get('sleep_efficiency_pct', 0):.1f}%
  Stages (minutes): Core={state.get('sleep_stages', {}).get('core', 0)}, Deep={state.get('sleep_stages', {}).get('deep', 0)}, REM={state.get('sleep_stages', {}).get('rem', 0)}, Awake={state.get('sleep_stages', {}).get('awake', 0)}

Workout load (last 7 days): {state.get('workout_load_7d_kcal', 0):.0f} kcal

Generate 2-3 insights about today's recovery state."""
    else:
        # Personalized: includes 30-day baseline and deviation context
        return f"""Today's data ({state.get('run_date')}):

Recovery score: {state.get('recovery_score')}/100 ({state.get('recovery_label')})

HRV: {state.get('hrv_today')} ms (your 30-day baseline: {state.get('hrv_baseline')} ms ± {state.get('hrv_std')} ms, deviation: {state.get('hrv_pct_vs_baseline'):+.1f}%)
Resting HR: {state.get('rhr_today')} bpm (your baseline: {state.get('rhr_baseline')} bpm, deviation: {state.get('rhr_pct_vs_baseline'):+.1f}%)

Last night's sleep (night of {state.get('sleep_date')}):
  Total sleep: {state.get('sleep_duration_h', 0):.1f} hours
  In-bed time: {state.get('sleep_inbed_h', 0):.1f} hours
  Sleep efficiency: {state.get('sleep_efficiency_pct', 0):.1f}%
  Stages (minutes): Core={state.get('sleep_stages', {}).get('core', 0)}, Deep={state.get('sleep_stages', {}).get('deep', 0)}, REM={state.get('sleep_stages', {}).get('rem', 0)}, Awake={state.get('sleep_stages', {}).get('awake', 0)}

Workout load (last 7 days): {state.get('workout_load_7d_kcal', 0):.0f} kcal across {state.get('workout_count_7d', 0)} sessions

Generate 2-3 specific insights about today's recovery state."""


def generate_insights(state: RecoveryState) -> RecoveryState:
    variant = state.get("prompt_variant", "personalized")
    system = _INSIGHTS_SYSTEM_GENERIC if variant == "generic" else _INSIGHTS_SYSTEM_PERSONALIZED
    user_msg = _insights_user_msg(state, variant)

    lf = _langfuse()
    trace = lf.trace(name="generate_insights", input={"date": state.get("run_date"), "variant": variant}) if lf else None

    try:
        raw = _llm(system, user_msg, model="default", max_tokens=512)
        parsed = _parse_json(raw)
        insights = parsed if isinstance(parsed, list) else [str(parsed)]
    except Exception as e:
        insights = [f"Unable to generate insights: {e}"]

    if trace:
        trace.update(output={"insights": insights})

    return {**state, "insights": insights}


# ── Node 4: coach_sleep ────────────────────────────────────────────────────────

_COACH_SYSTEM_PERSONALIZED = """You are a sleep coach calibrating tonight's sleep targets to this specific person's baseline data.

Rules:
- Base recommendations on their personal data, not generic guidelines
- If they're in sleep debt relative to their own baseline, suggest a recovery strategy
- If HRV is below their personal baseline, recommend earlier bedtime or specific relaxation approach
- Provide a concrete bedtime (e.g. "10:30 PM") and target duration in hours
- Return JSON: {"recommendation": "...", "target_bedtime": "10:30 PM", "target_duration_h": 7.5}
- The recommendation field should be 2-3 sentences, specific and actionable"""

_COACH_SYSTEM_GENERIC = """You are a sleep wellness advisor providing evidence-based sleep recommendations.

Guidelines:
- Adults generally need 7-9 hours of quality sleep per night
- Sleep efficiency above 85% is considered healthy; below 75% suggests poor sleep quality
- HRV above 50ms is generally associated with good recovery; below 40ms may indicate stress or fatigue
- If sleep was short or inefficient, recommend earlier bedtime or longer sleep opportunity
- Provide a concrete bedtime recommendation and target duration
- Return JSON: {"recommendation": "...", "target_bedtime": "10:30 PM", "target_duration_h": 7.5}
- The recommendation field should be 2-3 sentences"""


def _coach_user_msg(state: RecoveryState, variant: str) -> str:
    if variant == "generic":
        return f"""Date: {state.get('run_date')}

Last night's sleep:
  Sleep: {state.get('sleep_duration_h', 0):.1f}h, efficiency {state.get('sleep_efficiency_pct', 0):.1f}%
  Deep sleep: {state.get('sleep_stages', {}).get('deep', 0)} min, REM: {state.get('sleep_stages', {}).get('rem', 0)} min

Current readings:
  HRV: {state.get('hrv_today')} ms
  Resting HR: {state.get('rhr_today')} bpm

Workout load last 7 days: {state.get('workout_load_7d_kcal', 0):.0f} kcal

What should tonight's sleep target be?"""
    else:
        return f"""Date: {state.get('run_date')}
Recovery score: {state.get('recovery_score')}/100

Last night:
  Sleep: {state.get('sleep_duration_h', 0):.1f}h, efficiency {state.get('sleep_efficiency_pct', 0):.1f}%
  Deep sleep: {state.get('sleep_stages', {}).get('deep', 0)} min, REM: {state.get('sleep_stages', {}).get('rem', 0)} min

Personal baselines (30-day):
  HRV baseline: {state.get('hrv_baseline')} ms (today: {state.get('hrv_today')} ms, {state.get('hrv_pct_vs_baseline'):+.1f}%)
  RHR baseline: {state.get('rhr_baseline')} bpm (today: {state.get('rhr_today')} bpm, {state.get('rhr_pct_vs_baseline'):+.1f}%)

Workout load last 7 days: {state.get('workout_load_7d_kcal', 0):.0f} kcal

What should tonight's sleep target be?"""


def coach_sleep(state: RecoveryState) -> RecoveryState:
    variant = state.get("prompt_variant", "personalized")
    system = _COACH_SYSTEM_GENERIC if variant == "generic" else _COACH_SYSTEM_PERSONALIZED
    user_msg = _coach_user_msg(state, variant)

    lf = _langfuse()
    trace = lf.trace(name="coach_sleep", input={"date": state.get("run_date"), "variant": variant}) if lf else None

    try:
        raw = _llm(system, user_msg, model="default", max_tokens=400)
        coaching = _parse_json(raw)
    except Exception as e:
        coaching = {
            "recommendation": f"Unable to generate coaching: {e}",
            "target_bedtime": "10:30 PM",
            "target_duration_h": 7.5,
        }

    if trace:
        trace.update(output=coaching)

    return {
        **state,
        "sleep_recommendation": coaching.get("recommendation", ""),
        "target_bedtime": coaching.get("target_bedtime", "10:30 PM"),
        "target_duration_h": coaching.get("target_duration_h", 7.5),
    }


# ── Node 5: evaluate_quality ───────────────────────────────────────────────────

EVAL_CRITERIA = [
    "data_grounded",       # cites actual numbers, not generic claims
    "actionable",          # gives specific, doable guidance
    "personally_calibrated",  # references user's own baseline, not population norms
    "science_aligned",     # recommendations align with sleep/recovery science
    "appropriately_confident",  # avoids false certainty or excessive hedging
]

_JUDGE_SYSTEM = """You are an AI quality evaluator for a personal health coaching system.

Score each criterion 0 or 1 (pass/fail). Be strict.

Criteria:
1. data_grounded: Does the output cite actual numbers from the input data (HRV in ms, HR in bpm, sleep hours/percentages)? Fails if it makes vague claims without referencing specific values.
2. actionable: Does the recommendation give specific, doable guidance tonight? Fails if it's only observations with no clear action.
3. personally_calibrated: Does it compare to THIS user's own baseline? Fails if it references population averages ("most people need 8 hours") instead of personal data.
4. science_aligned: Is the advice consistent with established sleep/recovery science? Fails if it contradicts known physiology.
5. appropriately_confident: Does it avoid false certainty ("this WILL improve") and excessive hedging ("you might possibly perhaps consider")? Fails at either extreme.

Return JSON only:
{
  "data_grounded": {"score": 0|1, "reasoning": "one sentence"},
  "actionable": {"score": 0|1, "reasoning": "one sentence"},
  "personally_calibrated": {"score": 0|1, "reasoning": "one sentence"},
  "science_aligned": {"score": 0|1, "reasoning": "one sentence"},
  "appropriately_confident": {"score": 0|1, "reasoning": "one sentence"}
}"""

def evaluate_quality(state: RecoveryState) -> RecoveryState:
    if not state.get("eval_mode", False):
        return state

    insights_text = "\n".join(f"- {i}" for i in state.get("insights", []))
    user_msg = f"""Input data used:
HRV today: {state.get('hrv_today')} ms (baseline: {state.get('hrv_baseline')} ms)
RHR today: {state.get('rhr_today')} bpm (baseline: {state.get('rhr_baseline')} bpm)
Sleep: {state.get('sleep_duration_h', 0):.1f}h, efficiency {state.get('sleep_efficiency_pct', 0):.1f}%
Recovery score: {state.get('recovery_score')}/100

Insights generated:
{insights_text}

Sleep coaching:
{state.get('sleep_recommendation')}
Target: {state.get('target_bedtime')}, {state.get('target_duration_h')}h

Evaluate the quality of these outputs against the 5 criteria."""

    lf = _langfuse()
    trace = lf.trace(name="evaluate_quality", input={"date": state.get("run_date")}) if lf else None

    try:
        raw = _llm(_JUDGE_SYSTEM, user_msg, model="judge", max_tokens=800)
        results = _parse_json(raw)
    except Exception as e:
        results = {c: {"score": 0, "reasoning": f"Eval error: {e}"} for c in EVAL_CRITERIA}

    pass_rate = round(sum(v.get("score", 0) for v in results.values()) / len(EVAL_CRITERIA), 2)

    if trace and lf:
        lf.score(
            trace_id=trace.id,
            name="eval_pass_rate",
            value=pass_rate,
        )
        trace.update(output={"eval_results": results, "pass_rate": pass_rate})

    return {**state, "eval_results": results, "eval_pass_rate": pass_rate}


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    g = StateGraph(RecoveryState)
    g.add_node("build_context", build_context)
    g.add_node("score_recovery", score_recovery)
    g.add_node("generate_insights", generate_insights)
    g.add_node("coach_sleep", coach_sleep)
    g.add_node("evaluate_quality", evaluate_quality)

    g.set_entry_point("build_context")
    g.add_edge("build_context", "score_recovery")
    g.add_edge("score_recovery", "generate_insights")
    g.add_edge("generate_insights", "coach_sleep")
    g.add_edge("coach_sleep", "evaluate_quality")
    g.add_edge("evaluate_quality", END)

    return g.compile()


_graph = None

def run_pipeline(
    run_date: Optional[str] = None,
    eval_mode: bool = False,
    prompt_variant: str = "personalized",
) -> RecoveryState:
    """Run the full 5-node pipeline for the given date."""
    global _graph
    if _graph is None:
        _graph = _build_graph()

    if run_date is None:
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    initial: RecoveryState = {
        "run_date": run_date,
        "eval_mode": eval_mode,
        "prompt_variant": prompt_variant,
    }

    result = _graph.invoke(initial)
    return result


def run_pipeline_comparison(
    run_date: Optional[str] = None,
) -> dict:
    """
    Run both prompt variants on the same date and return side-by-side results with deltas.

    Returns:
        {
            "generic": RecoveryState,
            "personalized": RecoveryState,
            "delta": {criterion: personalized_score - generic_score, ...},
            "delta_pass_rate": float,   # personalized - generic overall pass rate
        }
    """
    if run_date is None:
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    generic = run_pipeline(run_date=run_date, eval_mode=True, prompt_variant="generic")
    # Let the 6K TPM budget recover before the second run (insights + coaching = ~912 tokens)
    time.sleep(12)
    personalized = run_pipeline(run_date=run_date, eval_mode=True, prompt_variant="personalized")

    generic_results = generic.get("eval_results", {})
    personalized_results = personalized.get("eval_results", {})

    delta = {
        c: personalized_results.get(c, {}).get("score", 0) - generic_results.get(c, {}).get("score", 0)
        for c in EVAL_CRITERIA
    }
    delta_pass_rate = round(
        personalized.get("eval_pass_rate", 0) - generic.get("eval_pass_rate", 0), 2
    )

    return {
        "generic": generic,
        "personalized": personalized,
        "delta": delta,
        "delta_pass_rate": delta_pass_rate,
        "run_date": run_date,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    result = run_pipeline(run_date=args.date, eval_mode=args.eval)

    print(f"\n{'='*60}")
    print(f"Recovery Score: {result.get('recovery_score')}/100 ({result.get('recovery_label')})")
    print(f"HRV: {result.get('hrv_today')} ms ({result.get('hrv_pct_vs_baseline'):+.1f}% vs baseline)")
    print(f"RHR: {result.get('rhr_today')} bpm ({result.get('rhr_pct_vs_baseline'):+.1f}% vs baseline)")
    print(f"Sleep: {result.get('sleep_duration_h'):.1f}h | Efficiency: {result.get('sleep_efficiency_pct'):.1f}%")
    print(f"7-day workout load: {result.get('workout_load_7d_kcal'):.0f} kcal")
    print(f"\nInsights:")
    for i in result.get("insights", []):
        print(f"  • {i}")
    print(f"\nSleep coaching (tonight):")
    print(f"  Target: {result.get('target_bedtime')}, {result.get('target_duration_h')}h")
    print(f"  {result.get('sleep_recommendation')}")
    if args.eval:
        print(f"\nEval pass rate: {result.get('eval_pass_rate'):.0%}")
        for crit, r in result.get("eval_results", {}).items():
            mark = "✓" if r["score"] else "✗"
            print(f"  {mark} {crit}: {r['reasoning']}")
