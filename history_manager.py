# history_manager.py
# Manages the permanent analysis history database and teacher feedback learning.
# ENHANCED: top_match columns, learning context for Granite, extended stats.

import os
import pandas as pd
from datetime import datetime

# ── File paths ────────────────────────────────────────────────────────────────

HISTORY_FILE = "data/analysis_history.csv"

# All columns — extended with top_matching_student and top_matching_assignment
# so reports and analytics can show who matched whom without re-running TF-IDF.
HISTORY_COLUMNS = [
    "timestamp",
    "student_id",
    "student_name",
    "assignment_title",
    "similarity_score",
    "risk_level",
    "teacher_decision",
    "teacher_notes",
    "granite_summary",
    "top_matching_student",      # NEW: name of the most-similar stored student
    "top_matching_assignment",   # NEW: assignment title of the top match
]


# ── Load / Save helpers ───────────────────────────────────────────────────────

def load_history() -> pd.DataFrame:
    """
    Load the full analysis history from CSV.
    Back-fills any new columns so older CSV files continue to work.
    """
    os.makedirs("data", exist_ok=True)
    try:
        df = pd.read_csv(HISTORY_FILE)
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[HISTORY_COLUMNS]
    except FileNotFoundError:
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def _save_history(df: pd.DataFrame) -> None:
    """Write the history DataFrame to CSV — always appends by re-saving the full frame."""
    os.makedirs("data", exist_ok=True)
    df.to_csv(HISTORY_FILE, index=False)


# ── Record a new analysis entry ───────────────────────────────────────────────

def record_analysis(
    student_id: str,
    student_name: str,
    assignment_title: str,
    similarity_score: float,
    risk_level: str,
    teacher_decision: str = "",
    teacher_notes: str = "",
    granite_summary: str = "",
    top_matching_student: str = "",
    top_matching_assignment: str = "",
) -> None:
    """
    Append one permanent analysis record to analysis_history.csv.
    Called after the teacher saves feedback so all fields are populated.
    Never overwrites previous records — always appends.
    """
    df = load_history()
    new_row = pd.DataFrame([{
        "timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M"),
        "student_id":              student_id,
        "student_name":            student_name,
        "assignment_title":        assignment_title,
        "similarity_score":        round(float(similarity_score), 4),
        "risk_level":              risk_level,
        "teacher_decision":        teacher_decision,
        "teacher_notes":           teacher_notes,
        "granite_summary":         granite_summary,
        "top_matching_student":    top_matching_student,
        "top_matching_assignment": top_matching_assignment,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    _save_history(df)


# ── Feedback statistics ───────────────────────────────────────────────────────

def get_feedback_stats(df: pd.DataFrame | None = None) -> dict:
    """
    Compute comprehensive teacher-decision and risk statistics from history.

    Returns a dict with keys:
        total, genuine, suspicious, confirmed, false_positive,
        high_risk, medium_risk, low_risk,
        avg_similarity, avg_confirmed_similarity, avg_fp_similarity,
        last_review, teacher_accuracy
    """
    if df is None:
        df = load_history()

    empty = {
        "total": 0, "genuine": 0, "suspicious": 0,
        "confirmed": 0, "false_positive": 0,
        "high_risk": 0, "medium_risk": 0, "low_risk": 0,
        "avg_similarity": 0.0,
        "avg_confirmed_similarity": 0.0,
        "avg_fp_similarity": 0.0,
        "last_review": "—",
        "teacher_accuracy": "—",
    }
    if df.empty:
        return empty

    decisions = df["teacher_decision"].str.strip()
    risk      = df["risk_level"].str.strip()
    scores    = pd.to_numeric(df["similarity_score"], errors="coerce")

    confirmed_mask = decisions == "Confirmed Plagiarism"
    fp_mask        = decisions == "False Positive"

    avg_confirmed = (scores[confirmed_mask].mean() * 100) if confirmed_mask.any() else 0.0
    avg_fp        = (scores[fp_mask].mean() * 100)        if fp_mask.any()        else 0.0

    reviewed = int((decisions != "").sum())
    # Teacher accuracy = confirmed / (confirmed + false_positive) * 100
    true_pos   = int(confirmed_mask.sum())
    false_pos  = int(fp_mask.sum())
    accuracy   = f"{true_pos / (true_pos + false_pos) * 100:.1f}%" if (true_pos + false_pos) > 0 else "—"

    # Last review timestamp
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    last_review = timestamps.max().strftime("%Y-%m-%d %H:%M") if not timestamps.empty else "—"

    return {
        "total":                    len(df),
        "genuine":                  int((decisions == "Genuine").sum()),
        "suspicious":               int((decisions == "Suspicious").sum()),
        "confirmed":                true_pos,
        "false_positive":           false_pos,
        "high_risk":                int(risk.str.contains("High",   na=False).sum()),
        "medium_risk":              int(risk.str.contains("Medium", na=False).sum()),
        "low_risk":                 int(risk.str.contains("Low",    na=False).sum()),
        "avg_similarity":           round(scores.mean() * 100, 1) if not scores.empty else 0.0,
        "avg_confirmed_similarity": round(avg_confirmed, 1),
        "avg_fp_similarity":        round(avg_fp, 1),
        "last_review":              last_review,
        "teacher_accuracy":         accuracy,
    }


# ── Feedback learning context for Granite ────────────────────────────────────

def get_learning_context(df: pd.DataFrame | None = None) -> str:
    """
    Build a short natural-language learning context string from historical decisions.
    This is injected into the Granite prompt so future analyses benefit from
    the teacher's accumulated feedback patterns.

    Returns an empty string if there is insufficient history (< 3 records).
    """
    if df is None:
        df = load_history()

    if len(df) < 3:
        return ""  # Not enough history to form meaningful patterns yet

    decisions = df["teacher_decision"].str.strip()
    scores    = pd.to_numeric(df["similarity_score"], errors="coerce")

    confirmed = df[decisions == "Confirmed Plagiarism"]
    genuine   = df[decisions == "Genuine"]
    fp        = df[decisions == "False Positive"]

    lines = ["HISTORICAL TEACHER DECISIONS (use as context only, do not override your analysis):"]

    if not confirmed.empty:
        avg_c = scores[decisions == "Confirmed Plagiarism"].mean() * 100
        lines.append(
            f"- {len(confirmed)} submission(s) previously marked as Confirmed Plagiarism "
            f"(average similarity: {avg_c:.1f}%)."
        )

    if not genuine.empty:
        avg_g = scores[decisions == "Genuine"].mean() * 100
        lines.append(
            f"- {len(genuine)} submission(s) marked as Genuine "
            f"(average similarity: {avg_g:.1f}%)."
        )

    if not fp.empty:
        avg_f = scores[decisions == "False Positive"].mean() * 100
        lines.append(
            f"- {len(fp)} submission(s) marked as False Positive "
            f"(average similarity: {avg_f:.1f}%). "
            f"The teacher noted these were incorrectly flagged by the system."
        )

    # Most common decision
    most_common = decisions[decisions != ""].value_counts()
    if not most_common.empty:
        lines.append(
            f"- The most frequent teacher decision overall is: {most_common.index[0]}."
        )

    return "\n".join(lines)


# ── Search / filter helpers ───────────────────────────────────────────────────

def search_history(
    df: pd.DataFrame,
    query: str = "",
    field: str = "student_name",
) -> pd.DataFrame:
    """Filter rows where *field* contains *query* (case-insensitive)."""
    if not query.strip():
        return df
    return df[df[field].astype(str).str.contains(query.strip(), case=False, na=False)]


def filter_by_risk(df: pd.DataFrame, risk_label: str) -> pd.DataFrame:
    """Return rows matching a specific risk label (partial, case-insensitive)."""
    return df[df["risk_level"].str.contains(risk_label, case=False, na=False)]


def filter_by_decision(df: pd.DataFrame, decision: str) -> pd.DataFrame:
    """Return rows where teacher_decision exactly matches *decision* (case-insensitive)."""
    return df[df["teacher_decision"].str.strip().str.lower() == decision.lower()]
