# dashboard.py
# Professional analytics dashboard rendered inside a Streamlit tab.
# ENHANCED: Plotly charts, IBM Blue palette, all 11 metric cards,
# 5 chart types, full search/filter with date range.

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False   # graceful fallback to st.bar_chart

from history_manager import load_history, get_feedback_stats

# ── IBM Blue colour palette ───────────────────────────────────────────────────
IBM_BLUE    = "#0f62fe"
IBM_PURPLE  = "#8a3ffc"
IBM_RED     = "#da1e28"
IBM_ORANGE  = "#ff832b"
IBM_GREEN   = "#198038"
IBM_TEAL    = "#009d9a"
IBM_GRAY    = "#525252"
IBM_YELLOW  = "#f1c21b"

RISK_COLOURS = {
    "High":   IBM_RED,
    "Medium": IBM_ORANGE,
    "Low":    IBM_GREEN,
}

DECISION_COLOURS = {
    "Confirmed Plagiarism": IBM_RED,
    "Suspicious":           IBM_ORANGE,
    "Genuine":              IBM_GREEN,
    "False Positive":       IBM_GRAY,
    "Pending":              IBM_YELLOW,
}


# ── Metric card helper ────────────────────────────────────────────────────────

def _card(col, label: str, value, colour: str = IBM_BLUE, icon: str = ""):
    """Render a professional IBM-styled metric card inside a Streamlit column."""
    col.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #e0e0e0;
                    border-left:4px solid {colour};border-radius:6px;
                    padding:16px 14px;text-align:center;
                    box-shadow:0 1px 3px rgba(0,0,0,.06);">
            <div style="font-size:24px;font-weight:700;color:{colour};
                        letter-spacing:-0.5px;">{value}</div>
            <div style="font-size:11px;color:#525252;margin-top:5px;
                        text-transform:uppercase;letter-spacing:.6px;">{icon} {label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Plotly / fallback chart helpers ──────────────────────────────────────────

def _bar(title: str, df_chart: pd.DataFrame, x: str, y: str,
         colour_map: dict | None = None, col=None):
    """Render a Plotly bar chart (or st.bar_chart fallback) inside *col* or st directly."""
    target = col if col else st
    target.markdown(f"**{title}**")
    if df_chart.empty:
        target.info("No data yet.")
        return
    if PLOTLY:
        colours = [colour_map.get(v, IBM_BLUE) for v in df_chart[x]] if colour_map else None
        fig = px.bar(
            df_chart, x=x, y=y,
            color_discrete_sequence=colours or [IBM_BLUE],
            template="simple_white",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            font_family="IBM Plex Sans, sans-serif",
            plot_bgcolor="#ffffff",
            height=280,
            showlegend=False,
            xaxis_title="", yaxis_title="Count",
        )
        fig.update_traces(marker_line_width=0)
        target.plotly_chart(fig, use_container_width=True)
    else:
        target.bar_chart(df_chart.set_index(x)[y])


def _line(title: str, df_chart: pd.DataFrame, x: str, y: str, col=None):
    """Render a Plotly line chart or st fallback."""
    target = col if col else st
    target.markdown(f"**{title}**")
    if df_chart.empty:
        target.info("No data yet.")
        return
    if PLOTLY:
        fig = px.line(
            df_chart, x=x, y=y,
            markers=True,
            color_discrete_sequence=[IBM_BLUE],
            template="simple_white",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            font_family="IBM Plex Sans, sans-serif",
            height=280,
            xaxis_title="", yaxis_title="Count",
        )
        target.plotly_chart(fig, use_container_width=True)
    else:
        target.line_chart(df_chart.set_index(x)[y])


# ── Main dashboard renderer ───────────────────────────────────────────────────

def render_dashboard():
    """
    Render the complete analytics dashboard.
    Call this inside a Streamlit tab context.
    """
    st.markdown(
        '<h2 style="color:#0f62fe;font-weight:700;letter-spacing:-0.5px;">'
        '📊 Analytics Dashboard</h2>',
        unsafe_allow_html=True,
    )
    st.caption("Live statistics from analysis_history.csv · refreshes on every page load")
    st.divider()

    df    = load_history()
    stats = get_feedback_stats(df)

    # ── Row 1: Primary KPI cards (4 columns) ─────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    _card(c1, "Total Assignments",    stats["total"],                          IBM_BLUE,   "📄")
    _card(c2, "Granite Analyses",     stats["total"],                          IBM_PURPLE, "🤖")
    _card(c3, "Teacher Reviews",
          stats["genuine"]+stats["suspicious"]+stats["confirmed"]+stats["false_positive"],
                                                                                IBM_TEAL,   "👨‍🏫")
    _card(c4, "Confirmed Plagiarism", stats["confirmed"],                       IBM_RED,    "🚩")

    st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

    # ── Row 2: Risk + decision breakdown (5 columns) ──────────────────────────
    r1, r2, r3, r4, r5 = st.columns(5)
    _card(r1, "High Risk",      stats["high_risk"],     IBM_RED,    "🚨")
    _card(r2, "Medium Risk",    stats["medium_risk"],   IBM_ORANGE, "⚠️")
    _card(r3, "Low Risk",       stats["low_risk"],      IBM_GREEN,  "✅")
    _card(r4, "False Positives",stats["false_positive"],IBM_GRAY,   "❌")
    _card(r5, "Suspicious",     stats["suspicious"],    IBM_YELLOW, "🔍")

    st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

    # ── Row 3: Metric cards (3 columns) ──────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    _card(m1, "Avg Similarity",        f"{stats['avg_similarity']}%",        IBM_BLUE,   "📊")
    _card(m2, "Avg Confirmed Sim.",     f"{stats['avg_confirmed_similarity']}%", IBM_RED, "📈")
    _card(m3, "Teacher Accuracy",       stats["teacher_accuracy"],            IBM_GREEN,  "🎯")

    if df.empty:
        st.divider()
        st.info("No analysis history yet. Complete a submission review to populate the dashboard.")
        return

    st.divider()

    # Prepare charting columns
    df = df.copy()
    df["similarity_pct"] = pd.to_numeric(df["similarity_score"], errors="coerce") * 100
    df["_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["month"] = df["_dt"].dt.to_period("M").astype(str)

    # ── Row 4: Risk distribution + Teacher decisions ──────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        risk_raw = df["risk_level"].str.extract(r"(High|Medium|Low)", expand=False).fillna("Unknown")
        rc = risk_raw.value_counts().reset_index()
        rc.columns = ["Risk Level", "Count"]
        cmap = {r: RISK_COLOURS.get(r, IBM_BLUE) for r in rc["Risk Level"]}
        _bar("Risk Distribution", rc, "Risk Level", "Count", colour_map=cmap, col=col_a)

    with col_b:
        dec_raw = df["teacher_decision"].replace("", "Pending")
        dc = dec_raw.value_counts().reset_index()
        dc.columns = ["Decision", "Count"]
        cmap2 = {d: DECISION_COLOURS.get(d, IBM_BLUE) for d in dc["Decision"]}
        _bar("Teacher Decision Distribution", dc, "Decision", "Count", colour_map=cmap2, col=col_b)

    st.divider()

    # ── Row 5: Similarity histogram + Monthly trend ───────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        bins   = list(range(0, 110, 10))
        labels = [f"{b}–{b+10}%" for b in bins[:-1]]
        df["sim_bucket"] = pd.cut(
            df["similarity_pct"].fillna(0), bins=bins, labels=labels, right=False
        )
        bc = df["sim_bucket"].value_counts().sort_index().reset_index()
        bc.columns = ["Similarity Range", "Count"]
        _bar("Similarity Distribution", bc, "Similarity Range", "Count", col=col_c)

    with col_d:
        monthly = df.groupby("month").size().reset_index(name="Count")
        monthly = monthly.rename(columns={"month": "Month"})
        _line("Monthly Submission Trend", monthly, "Month", "Count", col=col_d)

    st.divider()

    # ── Row 6: Assignment growth over time ────────────────────────────────────
    st.markdown("**Assignment Growth Over Time**")
    if not df["_dt"].isna().all():
        growth = df.sort_values("_dt").reset_index(drop=True)
        growth["cumulative"] = range(1, len(growth) + 1)
        growth["date"] = growth["_dt"].dt.date.astype(str)
        if PLOTLY:
            fig = px.area(
                growth, x="date", y="cumulative",
                color_discrete_sequence=[IBM_BLUE],
                template="simple_white",
                labels={"date": "", "cumulative": "Total Submissions"},
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=220,
                font_family="IBM Plex Sans, sans-serif",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            growth_chart = growth.set_index("date")["cumulative"]
            st.line_chart(growth_chart)
    else:
        st.info("Timestamps unavailable for growth chart.")

    st.divider()

    # ── Search & Filter section ───────────────────────────────────────────────
    st.markdown(
        '<h4 style="color:#0f62fe;">🔍 Search & Filter History</h4>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        search_query = st.text_input("Search name / ID / assignment", placeholder="e.g. Alice")
    with f2:
        risk_filter = st.selectbox("Filter by risk", ["All", "High Risk", "Medium Risk", "Low Risk"])
    with f3:
        decision_filter = st.selectbox(
            "Filter by decision",
            ["All", "Genuine", "Suspicious", "Confirmed Plagiarism", "False Positive", "Pending"],
        )

    date_filter = st.selectbox(
        "Date range", ["All time", "Last week", "Last month", "Last year"]
    )

    filtered = df.copy()

    if search_query.strip():
        mask = (
            filtered["student_name"].str.contains(search_query, case=False, na=False) |
            filtered["student_id"].str.contains(search_query, case=False, na=False) |
            filtered["assignment_title"].str.contains(search_query, case=False, na=False)
        )
        filtered = filtered[mask]

    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"].str.contains(
            risk_filter.replace(" Risk", ""), case=False, na=False)]

    if decision_filter != "All":
        if decision_filter == "Pending":
            filtered = filtered[filtered["teacher_decision"].str.strip() == ""]
        else:
            filtered = filtered[
                filtered["teacher_decision"].str.strip().str.lower() == decision_filter.lower()
            ]

    if date_filter != "All time":
        days = {"Last week": 7, "Last month": 30, "Last year": 365}[date_filter]
        cutoff = datetime.now() - timedelta(days=days)
        filtered = filtered[filtered["_dt"] >= cutoff]

    st.caption(f"Showing **{len(filtered)}** of **{len(df)}** records")

    display_cols = [
        "timestamp", "student_name", "student_id",
        "assignment_title", "similarity_pct", "risk_level",
        "teacher_decision", "top_matching_student",
    ]
    show_df = filtered[[c for c in display_cols if c in filtered.columns]].copy()
    if "similarity_pct" in show_df.columns:
        show_df["similarity_pct"] = show_df["similarity_pct"].round(1).astype(str) + "%"
    show_df.columns = [
        c.replace("similarity_pct", "Similarity")
         .replace("student_name", "Student")
         .replace("student_id", "ID")
         .replace("assignment_title", "Assignment")
         .replace("risk_level", "Risk")
         .replace("teacher_decision", "Decision")
         .replace("top_matching_student", "Top Match")
         .replace("timestamp", "Timestamp")
        for c in show_df.columns
    ]
    st.dataframe(show_df, use_container_width=True, hide_index=True)
