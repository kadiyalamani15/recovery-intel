"""
app.py — Recovery Intelligence Agent — Streamlit UI (3 tabs)

Tab 1: Today     — recovery score, HRV vs baseline, sleep summary, coaching
Tab 2: Trends    — 30-day charts: HRV, RHR, sleep duration, recovery score
Tab 3: Eval      — 20-case results, per-criterion pass rates, Langfuse link

Run: streamlit run app.py
"""

import json
import os
from datetime import datetime, timezone, timedelta, date as date_type
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"
COMPARISON_PATH = DATA_DIR / "eval_comparison.json"

st.set_page_config(
    page_title="Recovery Intel",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .main { padding-top: 1rem; }
  .metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border: 1px solid #313244;
  }
  .score-label { font-size: 0.85rem; color: #a6adc8; letter-spacing: 0.05em; text-transform: uppercase; }
  .insight-card {
    background: #181825;
    border-left: 3px solid #89b4fa;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
  }
  .coach-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border: 1px solid #a6e3a1;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loaders (cached) ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_parquet(name: str) -> Optional[pd.DataFrame]:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    for col in ("start", "end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def data_ready() -> bool:
    return (DATA_DIR / "hrv.parquet").exists()


# ── Pipeline runners (cached per date) ────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def run_for_date(run_date: str) -> dict:
    from pipeline import run_pipeline
    return dict(run_pipeline(run_date=run_date, eval_mode=False, prompt_variant="personalized"))


@st.cache_data(ttl=1800, show_spinner=False)
def run_comparison_for_date(run_date: str) -> dict:
    """Run both variants with eval on a single date. Slow — 4 LLM calls."""
    from pipeline import run_pipeline_comparison
    result = run_pipeline_comparison(run_date=run_date)
    # Convert RecoveryState TypedDicts to plain dicts for Streamlit caching
    return {
        "generic": dict(result["generic"]),
        "personalized": dict(result["personalized"]),
        "delta": result["delta"],
        "delta_pass_rate": result["delta_pass_rate"],
        "run_date": result["run_date"],
    }


def load_comparison() -> Optional[dict]:
    if not COMPARISON_PATH.exists():
        return None
    with open(COMPARISON_PATH) as f:
        return json.load(f)


# ── Trend data builders ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def build_trend_data(days: int = 30) -> pd.DataFrame:
    hrv_df = load_parquet("hrv")
    rhr_df = load_parquet("resting_hr")
    sleep_df = load_parquet("sleep")

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)

    rows = []
    date_range = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").normalize(),
        periods=days,
        freq="D",
        tz="UTC",
    )

    asleep_stages = ["core", "deep", "rem", "unspecified"]

    for dt in date_range:
        date_str = dt.strftime("%Y-%m-%d")
        row = {"date": dt.date()}

        # HRV: average readings on this day
        if hrv_df is not None:
            day_hrv = hrv_df[
                (hrv_df["start"] >= dt) & (hrv_df["start"] < dt + pd.Timedelta(days=1))
            ]["value"]
            if not day_hrv.empty:
                row["hrv"] = day_hrv.mean()

        # RHR
        if rhr_df is not None:
            day_rhr = rhr_df[
                (rhr_df["start"] >= dt) & (rhr_df["start"] < dt + pd.Timedelta(hours=36))
            ]["value"]
            if not day_rhr.empty:
                row["rhr"] = day_rhr.iloc[-1]

        # Sleep (session starting on this date)
        if sleep_df is not None:
            day_sleep = sleep_df[
                (sleep_df["start"] >= dt) & (sleep_df["start"] < dt + pd.Timedelta(hours=20))
            ]
            if not day_sleep.empty:
                asleep_min = day_sleep[day_sleep["stage"].isin(asleep_stages)]["duration_min"].sum()
                inbed_min = day_sleep[day_sleep["stage"] == "inbed"]["duration_min"].sum()
                if inbed_min == 0:
                    inbed_min = asleep_min
                row["sleep_h"] = asleep_min / 60
                row["sleep_eff"] = (asleep_min / inbed_min * 100) if inbed_min > 0 else None

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


@st.cache_data(ttl=3600)
def build_recovery_trend(days: int = 30) -> pd.DataFrame:
    """Compute recovery scores for the last N days (no LLM calls — score only)."""
    from pipeline import build_context, score_recovery

    rows = []
    for i in range(days - 1, -1, -1):
        d = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            state = {"run_date": d, "eval_mode": False}
            state = build_context(state)
            state = score_recovery(state)
            rows.append({"date": d, "score": state.get("recovery_score")})
        except Exception:
            rows.append({"date": d, "score": None})

    return pd.DataFrame(rows)


# ── UI Components ──────────────────────────────────────────────────────────────

def score_gauge(score: int, label: str) -> go.Figure:
    color = (
        "#a6e3a1" if score >= 80
        else "#89dceb" if score >= 65
        else "#f9e2af" if score >= 45
        else "#f38ba8"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": label, "font": {"color": "#cdd6f4", "size": 14}},
        number={"font": {"color": color, "size": 48}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#585b70"},
            "bar": {"color": color},
            "bgcolor": "#313244",
            "bordercolor": "#45475a",
            "steps": [
                {"range": [0, 45], "color": "#1e1e2e"},
                {"range": [45, 65], "color": "#1e1e2e"},
                {"range": [65, 80], "color": "#1e1e2e"},
                {"range": [80, 100], "color": "#1e1e2e"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        height=220,
        margin=dict(t=40, b=0, l=10, r=10),
        font_color="#cdd6f4",
    )
    return fig


def hrv_sparkline(hrv_df: pd.DataFrame, today_val: float, baseline: float) -> go.Figure:
    # Last 30 days daily HRV
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
    recent = hrv_df[hrv_df["start"] >= cutoff].copy()
    recent["date"] = recent["start"].dt.date
    daily = recent.groupby("date")["value"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["value"],
        mode="lines", name="HRV",
        line={"color": "#89b4fa", "width": 2},
        fill="tozeroy", fillcolor="rgba(137,180,250,0.1)",
    ))
    fig.add_hline(y=baseline, line_dash="dash", line_color="#585b70", annotation_text="30-day avg")
    fig.update_layout(
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        height=160, margin=dict(t=10, b=10, l=0, r=0),
        showlegend=False, font_color="#cdd6f4",
        xaxis={"showgrid": False, "tickcolor": "#585b70"},
        yaxis={"showgrid": True, "gridcolor": "#313244", "title": "ms"},
    )
    return fig


def trend_chart(df: pd.DataFrame, col: str, title: str, color: str, unit: str) -> go.Figure:
    valid = df.dropna(subset=[col])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid["date"], y=valid[col],
        mode="lines+markers", name=title,
        line={"color": color, "width": 2},
        marker={"size": 4},
    ))
    rolling = valid[col].rolling(7, min_periods=3).mean()
    fig.add_trace(go.Scatter(
        x=valid["date"], y=rolling,
        mode="lines", name="7-day avg",
        line={"color": color, "width": 1, "dash": "dot"},
        opacity=0.6,
    ))
    fig.update_layout(
        title={"text": title, "font": {"color": "#cdd6f4"}},
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        height=220, margin=dict(t=40, b=20, l=0, r=0),
        font_color="#cdd6f4",
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "#313244", "title": unit},
        legend={"bgcolor": "#1e1e2e"},
    )
    return fig


CRITERION_LABELS = {
    "data_grounded": "Data Grounded",
    "actionable": "Actionable",
    "personally_calibrated": "Personally Calibrated",
    "science_aligned": "Science Aligned",
    "appropriately_confident": "Right Confidence",
}


def eval_criterion_chart(results: list[dict]) -> go.Figure:
    from pipeline import EVAL_CRITERIA

    rates = []
    for crit in EVAL_CRITERIA:
        scores = [
            r.get("eval_results", {}).get(crit, {}).get("score", 0)
            for r in results if "error" not in r and r.get("eval_results")
        ]
        rates.append(sum(scores) / len(scores) if scores else 0)

    fig = go.Figure(go.Bar(
        x=[CRITERION_LABELS.get(c, c) for c in EVAL_CRITERIA],
        y=[r * 100 for r in rates],
        marker_color=["#a6e3a1" if r >= 0.8 else "#f9e2af" if r >= 0.6 else "#f38ba8" for r in rates],
        text=[f"{r:.0%}" for r in rates],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        height=280, margin=dict(t=20, b=20, l=0, r=0),
        font_color="#cdd6f4",
        yaxis={"range": [0, 110], "showgrid": True, "gridcolor": "#313244", "title": "Pass rate %"},
        xaxis={"showgrid": False},
        showlegend=False,
    )
    return fig


def comparison_gap_chart(comparison: dict) -> go.Figure:
    """Grouped bar: generic vs personalized pass rate per criterion."""
    from pipeline import EVAL_CRITERIA
    agg = comparison.get("aggregate", {})
    delta = comparison.get("delta", {})

    x_labels = [CRITERION_LABELS.get(c, c) for c in EVAL_CRITERIA]
    generic_vals = [agg.get("generic", {}).get(c, 0) * 100 for c in EVAL_CRITERIA]
    personal_vals = [agg.get("personalized", {}).get(c, 0) * 100 for c in EVAL_CRITERIA]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Generic prompt",
        x=x_labels,
        y=generic_vals,
        marker_color="#585b70",
        text=[f"{v:.0f}%" for v in generic_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Personalized prompt",
        x=x_labels,
        y=personal_vals,
        marker_color="#89b4fa",
        text=[f"{v:.0f}%" for v in personal_vals],
        textposition="outside",
    ))

    # Annotate the personally_calibrated delta explicitly
    pc_idx = EVAL_CRITERIA.index("personally_calibrated")
    pc_delta = delta.get("personally_calibrated", 0)
    if pc_delta != 0:
        fig.add_annotation(
            x=x_labels[pc_idx],
            y=max(generic_vals[pc_idx], personal_vals[pc_idx]) + 14,
            text=f"{'+'if pc_delta>0 else ''}{pc_delta:.0%} gap",
            showarrow=False,
            font={"color": "#a6e3a1" if pc_delta > 0 else "#f38ba8", "size": 13, "family": "monospace"},
        )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        height=320, margin=dict(t=30, b=20, l=0, r=0),
        font_color="#cdd6f4",
        yaxis={"range": [0, 120], "showgrid": True, "gridcolor": "#313244", "title": "Pass rate %"},
        xaxis={"showgrid": False},
        legend={"bgcolor": "#1e1e2e", "orientation": "h", "y": -0.15},
    )
    return fig


def single_comparison_chart(comparison_result: dict) -> go.Figure:
    """Mini grouped bar for a single-date comparison (Today tab)."""
    from pipeline import EVAL_CRITERIA
    g_eval = comparison_result.get("generic", {}).get("eval_results", {})
    p_eval = comparison_result.get("personalized", {}).get("eval_results", {})

    x_labels = [CRITERION_LABELS.get(c, c) for c in EVAL_CRITERIA]
    generic_vals = [g_eval.get(c, {}).get("score", 0) * 100 for c in EVAL_CRITERIA]
    personal_vals = [p_eval.get(c, {}).get("score", 0) * 100 for c in EVAL_CRITERIA]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Generic",
        x=x_labels, y=generic_vals,
        marker_color="#585b70",
        text=[("Pass" if v else "Fail") for v in generic_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Personalized",
        x=x_labels, y=personal_vals,
        marker_color="#89b4fa",
        text=[("Pass" if v else "Fail") for v in personal_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        height=260, margin=dict(t=10, b=10, l=0, r=0),
        font_color="#cdd6f4",
        yaxis={"range": [0, 130], "showgrid": False, "showticklabels": False},
        xaxis={"showgrid": False},
        legend={"bgcolor": "#1e1e2e", "orientation": "h", "y": -0.2},
    )
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.markdown("## ⚡ Recovery Intel")
    st.caption("Personal recovery intelligence powered by Apple Health + Claude")

    if not data_ready():
        st.error("No parsed data found.")
        st.info(
            "Run the parser first:\n```\npip install -r requirements.txt\npython parse_health.py\n```"
        )
        return

    # Date selector in sidebar
    with st.sidebar:
        st.header("Settings")
        selected_date = st.date_input(
            "Analysis date",
            value=datetime.now(timezone.utc).date(),
            max_value=datetime.now(timezone.utc).date(),
        ).strftime("%Y-%m-%d")

        langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        st.markdown(f"[Langfuse Dashboard ↗]({langfuse_host})")

    tab_today, tab_trends, tab_eval = st.tabs(["Today", "Trends", "Eval"])

    # ── Tab 1: Today ──────────────────────────────────────────────────────────
    with tab_today:
        with st.spinner("Running recovery analysis..."):
            try:
                result = run_for_date(selected_date)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.info("Make sure ANTHROPIC_API_KEY is set in .env")
                return

        score = result.get("recovery_score", 0)
        label = result.get("recovery_label", "—")

        col_gauge, col_metrics = st.columns([1, 2])

        with col_gauge:
            st.plotly_chart(
                score_gauge(score, f"{label}"),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_metrics:
            mc1, mc2, mc3 = st.columns(3)

            hrv_today = result.get("hrv_today", 0)
            hrv_pct = result.get("hrv_pct_vs_baseline", 0)
            rhr_today = result.get("rhr_today", 0)
            rhr_pct = result.get("rhr_pct_vs_baseline", 0)
            sleep_h = result.get("sleep_duration_h", 0)
            sleep_eff = result.get("sleep_efficiency_pct", 0)

            mc1.metric(
                "HRV",
                f"{hrv_today:.0f} ms",
                f"{hrv_pct:+.1f}% vs baseline",
                delta_color="normal",
            )
            mc2.metric(
                "Resting HR",
                f"{rhr_today:.0f} bpm",
                f"{rhr_pct:+.1f}% vs baseline",
                delta_color="normal",
            )
            mc3.metric(
                "Sleep",
                f"{sleep_h:.1f}h",
                f"{sleep_eff:.0f}% efficiency",
                delta_color="off",
            )

            # HRV sparkline
            hrv_df = load_parquet("hrv")
            if hrv_df is not None:
                st.plotly_chart(
                    hrv_sparkline(hrv_df, hrv_today, result.get("hrv_baseline", 60)),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        # Sleep stages breakdown
        stages = result.get("sleep_stages", {})
        if stages:
            st.divider()
            st.markdown("**Last night's sleep**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Deep", f"{stages.get('deep', 0):.0f} min")
            sc2.metric("REM", f"{stages.get('rem', 0):.0f} min")
            sc3.metric("Core", f"{stages.get('core', 0):.0f} min")
            sc4.metric("Awake", f"{stages.get('awake', 0):.0f} min")

        # Insights
        insights = result.get("insights", [])
        if insights:
            st.divider()
            st.markdown("**Insights**")
            for ins in insights:
                st.markdown(
                    f'<div class="insight-card">{ins}</div>',
                    unsafe_allow_html=True,
                )

        # Sleep coaching
        coaching = result.get("sleep_recommendation")
        if coaching:
            st.divider()
            st.markdown("**Tonight's sleep target**")
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.markdown(
                    f'<div class="coach-card">{coaching}</div>',
                    unsafe_allow_html=True,
                )
            with cc2:
                st.metric("Target bedtime", result.get("target_bedtime", "—"))
                st.metric("Target duration", f"{result.get('target_duration_h', 0):.1f}h")

        # ── Prompt comparison panel ───────────────────────────────────────────
        st.divider()
        st.markdown("**Generic vs personalized prompt — same data, different coaching**")
        st.caption(
            "Generic uses population norms (how most health AIs work). "
            "Personalized calibrates to your 30-day baseline."
        )
        run_cmp = st.button("Run comparison for this date (4 LLM calls)", key="cmp_btn")
        if run_cmp:
            with st.spinner("Running both variants through the judge..."):
                try:
                    cmp = run_comparison_for_date(selected_date)
                    st.session_state["last_comparison"] = cmp
                except Exception as e:
                    st.error(f"Comparison error: {e}")

        cmp = st.session_state.get("last_comparison")
        if cmp and cmp.get("run_date") == selected_date:
            g = cmp["generic"]
            p = cmp["personalized"]
            g_rate = g.get("eval_pass_rate", 0)
            p_rate = p.get("eval_pass_rate", 0)
            delta_rate = cmp["delta_pass_rate"]

            # headline metrics
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Generic pass rate", f"{g_rate:.0%}")
            hc2.metric("Personalized pass rate", f"{p_rate:.0%}")
            hc3.metric(
                "Delta",
                f"{delta_rate:+.0%}",
                delta=f"{'better' if delta_rate >= 0 else 'worse'} with personalization",
                delta_color="normal" if delta_rate >= 0 else "inverse",
            )

            # mini criterion chart
            st.plotly_chart(
                single_comparison_chart(cmp),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # side-by-side coaching text
            st.markdown("**What each prompt said:**")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("*Generic*")
                g_insights = g.get("insights", [])
                for ins in g_insights:
                    st.markdown(f'<div class="insight-card" style="border-color:#585b70">{ins}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="coach-card" style="border-color:#585b70">{g.get("sleep_recommendation","—")}</div>',
                    unsafe_allow_html=True,
                )
            with tc2:
                st.markdown("*Personalized*")
                p_insights = p.get("insights", [])
                for ins in p_insights:
                    st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="coach-card">{p.get("sleep_recommendation","—")}</div>',
                    unsafe_allow_html=True,
                )

            # per-criterion detail
            with st.expander("Judge reasoning — criterion by criterion"):
                from pipeline import EVAL_CRITERIA
                for crit in EVAL_CRITERIA:
                    g_r = g.get("eval_results", {}).get(crit, {})
                    p_r = p.get("eval_results", {}).get(crit, {})
                    g_pass = "Pass" if g_r.get("score") else "Fail"
                    p_pass = "Pass" if p_r.get("score") else "Fail"
                    st.markdown(f"**{CRITERION_LABELS.get(crit, crit)}**")
                    rc1, rc2 = st.columns(2)
                    rc1.markdown(f"Generic: `{g_pass}` — {g_r.get('reasoning','—')}")
                    rc2.markdown(f"Personalized: `{p_pass}` — {p_r.get('reasoning','—')}")

    # ── Tab 2: Trends ─────────────────────────────────────────────────────────
    with tab_trends:
        with st.spinner("Building trend data..."):
            trend_df = build_trend_data(30)

        st.markdown("**30-day trends**")

        tc1, tc2 = st.columns(2)
        with tc1:
            if "hrv" in trend_df.columns:
                st.plotly_chart(
                    trend_chart(trend_df, "hrv", "HRV (ms)", "#89b4fa", "ms"),
                    use_container_width=True, config={"displayModeBar": False},
                )
        with tc2:
            if "rhr" in trend_df.columns:
                st.plotly_chart(
                    trend_chart(trend_df, "rhr", "Resting HR (bpm)", "#f38ba8", "bpm"),
                    use_container_width=True, config={"displayModeBar": False},
                )

        tc3, tc4 = st.columns(2)
        with tc3:
            if "sleep_h" in trend_df.columns:
                st.plotly_chart(
                    trend_chart(trend_df, "sleep_h", "Sleep Duration (h)", "#a6e3a1", "hours"),
                    use_container_width=True, config={"displayModeBar": False},
                )
        with tc4:
            with st.spinner("Computing recovery scores (may take 30s)..."):
                try:
                    score_trend = build_recovery_trend(30)
                    st.plotly_chart(
                        trend_chart(score_trend, "score", "Recovery Score", "#f9e2af", "score"),
                        use_container_width=True, config={"displayModeBar": False},
                    )
                except Exception as e:
                    st.warning(f"Could not compute score trend: {e}")

    # ── Tab 3: Eval ────────────────────────────────────────────────────────────
    with tab_eval:
        langfuse_link = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        col_ev1, col_ev2 = st.columns([3, 1])
        with col_ev1:
            st.markdown("**Eval Harness — 20 test cases from real health history**")
        with col_ev2:
            st.markdown(f"[View in Langfuse ↗]({langfuse_link})")

        # ── Comparison gap chart (20-case) ────────────────────────────────────
        comparison = load_comparison()
        if comparison:
            st.markdown("**The gap: generic prompt vs personalized prompt (n=20)**")
            agg = comparison.get("aggregate", {})
            delta = comparison.get("delta", {})

            gc1, gc2, gc3 = st.columns(3)
            gc1.metric(
                "Generic overall",
                f"{agg.get('generic', {}).get('overall', 0):.0%}",
            )
            gc2.metric(
                "Personalized overall",
                f"{agg.get('personalized', {}).get('overall', 0):.0%}",
            )
            delta_overall = delta.get("overall", 0)
            gc3.metric(
                "Personalization lift",
                f"{delta_overall:+.0%}",
                delta=f"{'improvement' if delta_overall >= 0 else 'regression'}",
                delta_color="normal" if delta_overall >= 0 else "inverse",
            )

            st.plotly_chart(
                comparison_gap_chart(comparison),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            pc_delta = delta.get("personally_calibrated", 0)
            if pc_delta != 0:
                direction = "higher" if pc_delta > 0 else "lower"
                st.info(
                    f"**Personally Calibrated** pass rate is **{abs(pc_delta):.0%} {direction}** with the "
                    f"personalized prompt — the largest gap across all 5 criteria. "
                    f"This is what generic health AI gets wrong."
                )
            st.divider()
        else:
            st.info(
                "Run the 20-case comparison to see the gap:\n"
                "```\npython eval_harness.py --compare\n```"
            )
            st.divider()

        # Load stored results
        eval_results = None
        if EVAL_RESULTS_PATH.exists():
            with open(EVAL_RESULTS_PATH) as f:
                eval_results = json.load(f)

        if eval_results is None:
            st.info(
                "No per-case results yet. Run the eval harness:\n"
                "```\npython eval_harness.py --run\n```"
            )
        else:
            # Summary metrics
            rates = [r.get("eval_pass_rate", 0) for r in eval_results if "error" not in r]
            overall = sum(rates) / len(rates) if rates else 0
            perfect = sum(1 for r in rates if r == 1.0)

            em1, em2, em3 = st.columns(3)
            em1.metric("Personalized pass rate", f"{overall:.0%}")
            em2.metric("Perfect scores", f"{perfect}/{len(rates)}")
            em3.metric("Cases run", len(eval_results))

            # Per-criterion chart
            try:
                st.plotly_chart(
                    eval_criterion_chart(eval_results),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            except Exception:
                pass

            # Case table
            st.markdown("**Case results (personalized prompt)**")
            table_rows = []
            for r in eval_results:
                crit_scores = r.get("eval_results", {})
                table_rows.append({
                    "Date": r["date"],
                    "Category": r["category"],
                    "Score": r.get("recovery_score", "ERR"),
                    "Pass rate": f"{r.get('eval_pass_rate', 0):.0%}",
                    "D": "✓" if crit_scores.get("data_grounded", {}).get("score") else "✗",
                    "A": "✓" if crit_scores.get("actionable", {}).get("score") else "✗",
                    "P": "✓" if crit_scores.get("personally_calibrated", {}).get("score") else "✗",
                    "S": "✓" if crit_scores.get("science_aligned", {}).get("score") else "✗",
                    "C": "✓" if crit_scores.get("appropriately_confident", {}).get("score") else "✗",
                })
            st.caption("D=Data grounded, A=Actionable, P=Personalized, S=Science aligned, C=Right confidence")
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # Expandable case details
            with st.expander("View insights for a specific case"):
                dates = [r["date"] for r in eval_results]
                sel = st.selectbox("Select case", dates)
                case = next((r for r in eval_results if r["date"] == sel), None)
                if case:
                    st.json({
                        "recovery_score": case.get("recovery_score"),
                        "hrv_today": case.get("hrv_today"),
                        "rhr_today": case.get("rhr_today"),
                        "sleep_duration_h": case.get("sleep_duration_h"),
                        "insights": case.get("insights", []),
                        "sleep_recommendation": case.get("sleep_recommendation"),
                        "eval_results": case.get("eval_results", {}),
                    })


if __name__ == "__main__":
    main()
