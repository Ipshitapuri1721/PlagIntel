# analytics.py
# PlagIntel Analytics page — teacher performance, risk/similarity trends,
# highest similarity cases, most-copied assignments, recent reviews.
# Rendered inside the "Analytics" Streamlit tab.

import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

from history_manager import load_history, get_feedback_stats
from plagiarism_engine import load_submissions

# ── IBM palette (mirrors dashboard.py) ───────────────────────────────────────
IBM_BLUE   = "#0f62fe"
IBM_RED    = "#da1e28"
IBM_ORANGE = "#ff832b"
IBM_GREEN  = "#198038"
IBM_PURPLE = "#8a3ffc"
IBM_GRAY   = "#525252"


# ── Section header helper ─────────────────────────────────────────────────────

def _section(title: str):
    st.markdown(
        f'<h4 style="color:#0f62fe;font-weight:700;margin-top:8px;">{title}</h4>',
        unsafe_allow_html=True,
    )


# ── Mini stat card ────────────────────────────────────────────────────────────

def _mini_card(col, label: str, value, colour: str = IBM_BLUE):
    col.markdown(
        f"""<div style="background:#fff;border:1px solid #e0e0e0;border-left:4px solid {colour};
                        border-radius:6px;padding:12px 14px;text-align:center;
                        box-shadow:0 1px 3px rgba(0,0,0,.05);">
                <div style="font-size:20px;font-weight:700;color:{colour};">{value}</div>
                <div style="font-size:11px;color:#525252;margin-top:4px;
                            text-transform:uppercase;letter-spacing:.5px;">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# ── Plotly bar helper ─────────────────────────────────────────────────────────

def _pbar(df_c: pd.DataFrame, x: str, y: str, colour: str = IBM_BLUE, height: int = 300):
    if PLOTLY:
        fig = px.bar(
            df_c, x=x, y=y,
            color_discrete_sequence=[colour],
            template="simple_white",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=height, showlegend=False,
            xaxis_title="", yaxis_title="",
            font_family="IBM Plex Sans, sans-serif",
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df_c.set_index(x)[y])


# ── Main analytics renderer ───────────────────────────────────────────────────

def render_analytics():
    """Render the complete analytics page inside a Streamlit tab."""

    st.markdown(
        '<h2 style="color:#0f62fe;font-weight:700;letter-spacing:-0.5px;">'
        '📈 Analytics</h2>',
        unsafe_allow_html=True,
    )
    st.caption("Deep-dive into patterns, trends, and performance metrics")
    st.divider()

    df      = load_history()
    df_subs = load_submissions()
    stats   = get_feedback_stats(df)

    if df.empty:
        st.info(
            "No analysis history yet. Complete at least one full review workflow "
            "to populate the analytics page."
        )
        return

    # Prepare numeric similarity column
    df = df.copy()
    df["sim_pct"] = pd.to_numeric(df["similarity_score"], errors="coerce") * 100
    df["_dt"]     = pd.to_datetime(df["timestamp"], errors="coerce")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Teacher Performance
    # ══════════════════════════════════════════════════════════════════════════

    _section("👨‍🏫 Teacher Performance")

    tp1, tp2, tp3, tp4 = st.columns(4)
    _mini_card(tp1, "Total Reviews",    stats["genuine"]+stats["suspicious"]+stats["confirmed"]+stats["false_positive"], IBM_BLUE)
    _mini_card(tp2, "Confirmed Cases",  stats["confirmed"],    IBM_RED)
    _mini_card(tp3, "False Positives",  stats["false_positive"], IBM_GRAY)
    _mini_card(tp4, "Accuracy",         stats["teacher_accuracy"], IBM_GREEN)

    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

    # Decision breakdown pie / bar
    decisions = df["teacher_decision"].replace("", "Pending")
    dc = decisions.value_counts().reset_index()
    dc.columns = ["Decision", "Count"]
    if not dc.empty and PLOTLY:
        colour_seq = [IBM_RED, IBM_ORANGE, IBM_GREEN, IBM_GRAY, IBM_PURPLE]
        fig = px.pie(
            dc, names="Decision", values="Count",
            color_discrete_sequence=colour_seq,
            hole=0.45,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            font_family="IBM Plex Sans, sans-serif",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    elif not dc.empty:
        st.bar_chart(dc.set_index("Decision")["Count"])

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Risk & Similarity Trends
    # ══════════════════════════════════════════════════════════════════════════

    _section("📊 Risk & Similarity Trends")

    tr1, tr2 = st.columns(2)

    with tr1:
        st.markdown("**Average Similarity Over Time**")
        if not df["_dt"].isna().all():
            trend = (
                df.dropna(subset=["_dt"])
                  .sort_values("_dt")
                  .assign(month=lambda d: d["_dt"].dt.to_period("M").astype(str))
                  .groupby("month")["sim_pct"]
                  .mean()
                  .reset_index()
            )
            trend.columns = ["Month", "Avg Similarity %"]
            if PLOTLY:
                fig = px.line(
                    trend, x="Month", y="Avg Similarity %",
                    markers=True,
                    color_discrete_sequence=[IBM_BLUE],
                    template="simple_white",
                )
                fig.update_layout(
                    height=260, margin=dict(l=0,r=0,t=10,b=0),
                    font_family="IBM Plex Sans, sans-serif",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(trend.set_index("Month"))
        else:
            st.info("No timestamp data available.")

    with tr2:
        st.markdown("**Risk Level Counts by Month**")
        if not df["_dt"].isna().all():
            df["risk_clean"] = df["risk_level"].str.extract(r"(High|Medium|Low)", expand=False).fillna("Unknown")
            df["month"]      = df["_dt"].dt.to_period("M").astype(str)
            risk_trend = (
                df.groupby(["month", "risk_clean"])
                  .size()
                  .reset_index(name="Count")
            )
            if PLOTLY:
                colour_map = {"High": IBM_RED, "Medium": IBM_ORANGE, "Low": IBM_GREEN}
                fig = px.bar(
                    risk_trend, x="month", y="Count", color="risk_clean",
                    barmode="group",
                    color_discrete_map=colour_map,
                    template="simple_white",
                )
                fig.update_layout(
                    height=260, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis_title="", yaxis_title="Count",
                    legend_title="Risk",
                    font_family="IBM Plex Sans, sans-serif",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(risk_trend.pivot(index="month", columns="risk_clean", values="Count"))
        else:
            st.info("No timestamp data available.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Highest Similarity Cases
    # ══════════════════════════════════════════════════════════════════════════

    _section("🚨 Highest Similarity Cases")

    top_cases = df.nlargest(10, "sim_pct")[
        ["timestamp", "student_name", "student_id", "assignment_title",
         "sim_pct", "risk_level", "teacher_decision", "top_matching_student"]
    ].copy()
    top_cases["sim_pct"] = top_cases["sim_pct"].round(1).astype(str) + "%"
    top_cases.columns = [
        "Timestamp", "Student", "ID", "Assignment",
        "Similarity", "Risk", "Decision", "Matched Against",
    ]
    st.dataframe(top_cases, use_container_width=True, hide_index=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Most Frequently Matched Assignments
    # ══════════════════════════════════════════════════════════════════════════

    _section("📋 Most Frequently Matched Assignments")

    if "top_matching_assignment" in df.columns:
        freq = (
            df["top_matching_assignment"]
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )
        freq.columns = ["Assignment", "Times Matched"]
        if not freq.empty:
            _pbar(freq, "Assignment", "Times Matched", IBM_PURPLE)
        else:
            st.info("No top-match assignment data recorded yet.")
    else:
        st.info("Top match data not yet available — submit and review assignments to populate.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Most Common Assignment Topics
    # ══════════════════════════════════════════════════════════════════════════

    _section("📚 Most Common Assignment Topics")

    topics = df["assignment_title"].value_counts().head(10).reset_index()
    topics.columns = ["Topic", "Submissions"]
    if not topics.empty:
        _pbar(topics, "Topic", "Submissions", IBM_TEAL := "#009d9a")
    else:
        st.info("No assignment topic data available.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Recent Reviews
    # ══════════════════════════════════════════════════════════════════════════

    _section("🕐 Recent Reviews")

    recent = df.sort_values("_dt", ascending=False).head(15)[
        ["timestamp", "student_name", "assignment_title",
         "sim_pct", "risk_level", "teacher_decision"]
    ].copy()
    recent["sim_pct"] = recent["sim_pct"].round(1).astype(str) + "%"
    recent.columns = ["Timestamp", "Student", "Assignment", "Similarity", "Risk", "Decision"]
    st.dataframe(recent, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Similarity Distribution (confirmed vs genuine)
    # ══════════════════════════════════════════════════════════════════════════

    st.divider()
    _section("🔬 Similarity Distribution by Decision")

    confirmed_df = df[df["teacher_decision"] == "Confirmed Plagiarism"]["sim_pct"].dropna()
    genuine_df   = df[df["teacher_decision"] == "Genuine"]["sim_pct"].dropna()

    if PLOTLY and (not confirmed_df.empty or not genuine_df.empty):
        import plotly.figure_factory as ff
        data, labels, colours = [], [], []
        if not confirmed_df.empty:
            data.append(confirmed_df.tolist())
            labels.append("Confirmed Plagiarism")
            colours.append(IBM_RED)
        if not genuine_df.empty:
            data.append(genuine_df.tolist())
            labels.append("Genuine")
            colours.append(IBM_GREEN)
        if len(data) >= 1 and all(len(d) >= 2 for d in data):
            try:
                fig = ff.create_distplot(data, labels, colors=colours, bin_size=5.0,
                                         show_rug=False)
                fig.update_layout(
                    height=280, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis_title="Similarity %",
                    font_family="IBM Plex Sans, sans-serif",
                    template="simple_white",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Not enough data for distribution plot yet.")
        else:
            st.info("Need at least 2 records per decision type for distribution chart.")
    else:
        st.info("Submit and review more assignments to see similarity distributions.")
