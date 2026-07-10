# app.py
# PlagIntel — AI-Driven Plagiarism Intelligence for Assignments
# ENHANCED: 3-tab layout (Analyse / Dashboard / Analytics), IBM Blue UI,
# feedback learning context injected into Granite, PDF download, sidebar
# last-review stat, top-matching fields stored in history, full workflow preserved.

import streamlit as st
import pandas as pd
from datetime import datetime
import io

from plagiarism_engine import (
    compare_with_stored,
    calculate_overall_risk,
    add_new_submission,
    load_submissions,
    highlight_matches,
    run_hybrid_analysis,    # NEW: hybrid detection engine
)
from granite_explainer import explain_with_granite, first_submission_result
from history_manager import (
    load_history,
    record_analysis,
    get_feedback_stats,
    get_learning_context,      # NEW: injects past decisions into Granite prompt
)
from report_generator import (
    generate_txt_report,
    generate_csv_report,
    generate_pdf_report,       # NEW: PDF report
    safe_filename,
)
from dashboard  import render_dashboard
from analytics  import render_analytics   # NEW: dedicated analytics tab


# ══════════════════════════════════════════════════════════════════════════════
# File extraction helper
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_file(uploaded_file) -> tuple[str, str]:
    """
    Extract plain text from a Streamlit UploadedFile object.

    Supports TXT, PDF (via pypdf), and DOCX (via python-docx).
    Uses only pure-Python libraries — no native DLLs, no CUDA, no PyMuPDF/fitz.

    Returns
    -------
    (text, error_message)
        text          — extracted text string (empty string on failure)
        error_message — human-readable error string, or "" on success
    """
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

    # ── TXT ───────────────────────────────────────────────────────────────────
    if file_ext == "txt":
        try:
            return uploaded_file.read().decode("utf-8"), ""
        except UnicodeDecodeError:
            try:
                uploaded_file.seek(0)
                return uploaded_file.read().decode("latin-1"), ""
            except Exception:
                return "", "Unable to read the TXT file. Please ensure it is plain text."

    # ── PDF ───────────────────────────────────────────────────────────────────
    elif file_ext == "pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            pages  = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text.strip())
            text = "\n\n".join(pages).strip()
            if not text:
                return "", (
                    "This PDF appears to contain scanned images with no selectable text. "
                    "OCR software would be required to extract text from it."
                )
            return text, ""
        except Exception as exc:
            return "", f"Unable to read the PDF. ({type(exc).__name__})"

    # ── DOCX ──────────────────────────────────────────────────────────────────
    elif file_ext == "docx":
        try:
            from docx import Document
            doc  = Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return text, ""
        except Exception as exc:
            return "", f"The uploaded DOCX file could not be read. ({type(exc).__name__})"

    # ── Unsupported ───────────────────────────────────────────────────────────
    else:
        return "", f"Unsupported file format: .{file_ext}"



# ══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PlagIntel",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global IBM Blue CSS injection ─────────────────────────────────────────────
st.markdown("""
<style>
    /* IBM Blue accent on Streamlit default elements */
    .stButton > button[kind="primary"] {
        background-color: #0f62fe !important;
        border-color: #0f62fe !important;
        color: #ffffff !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        letter-spacing: .3px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0353e9 !important;
        border-color: #0353e9 !important;
    }
    /* Tab label styling */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 14px;
        letter-spacing: .2px;
    }
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #f4f4f4 !important;
    }
    /* Divider colour */
    hr { border-color: #e0e0e0 !important; }
    /* Main header */
    h1 { color: #0f62fe !important; letter-spacing: -0.5px !important; }
    h2 { color: #161616 !important; }
    h3 { color: #393939 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    """Initialise all session-state variables on first load."""
    defaults = {
        "results":          None,   # full compare_with_stored list
        "overall_risk":     None,   # calculate_overall_risk summary dict
        "granite_explain":  None,   # Granite case report string
        "hybrid_result":    None,   # NEW: hybrid analysis result dict
        "step":             1,      # current agent workflow step (1–6)
        "new_text":         "",     # submission text under analysis
        "feedback_saved":   False,  # True after teacher saves decision
        "feedback_log":     [],     # in-session feedback log
        # student info held across rerun so Steps 5 & 6 can read it
        "s_student_id":     "",
        "s_student_name":   "",
        "s_assign_title":   "",
        "s_submit_time":    "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80,
    )
    st.markdown(
        '<div style="font-size:18px;font-weight:700;color:#0f62fe;margin:4px 0 2px;">'
        'PlagIntel</div>'
        '<div style="font-size:11px;color:#525252;">AI-Driven Plagiarism Intelligence<br>'
        'IBM Granite · watsonx.ai</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # IBM services list
    st.markdown("""
**IBM Services Used**
- 🤖 IBM Granite
- ☁️ watsonx.ai Runtime
- 🎓 watsonx.ai Studio
- 🔗 Watsonx Orchestrate
""")

    st.divider()

    # Live database statistics
    df_stored  = load_submissions()
    df_history = load_history()
    stats      = get_feedback_stats(df_history)

    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:#0f62fe;'
        'text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">'
        '📊 Database Statistics</div>',
        unsafe_allow_html=True,
    )

    sa, sb = st.columns(2)
    sa.metric("Submissions", len(df_stored))
    sb.metric("Reviews",
              stats["genuine"] + stats["suspicious"] +
              stats["confirmed"] + stats["false_positive"])

    sc, sd = st.columns(2)
    sc.metric("Confirmed", stats["confirmed"])
    sd.metric("False Pos.", stats["false_positive"])

    # Last review
    st.caption(f"🕐 Last review: {stats['last_review']}")

    st.divider()

    # Stored submissions viewer
    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:#0f62fe;'
        'text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">'
        '📂 Stored Submissions</div>',
        unsafe_allow_html=True,
    )
    if df_stored.empty:
        st.info("No previous submissions found.")
    else:
        with st.expander(f"View all {len(df_stored)} submissions"):
            view_cols = [c for c in
                         ["student_id", "student_name", "assignment_title"]
                         if c in df_stored.columns]
            st.dataframe(df_stored[view_cols], use_container_width=True, hide_index=True)

    st.divider()

    # In-session feedback log
    if st.session_state["feedback_log"]:
        st.markdown(
            '<div style="font-size:13px;font-weight:700;color:#0f62fe;'
            'text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">'
            '🗒️ Session Decisions</div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"{len(st.session_state['feedback_log'])} decisions this session"):
            for entry in reversed(st.session_state["feedback_log"]):
                st.markdown(
                    f"**{entry['timestamp']}**  \n"
                    f"Student: {entry['student_name']} · {entry['risk_level']}  \n"
                    f"Decision: `{entry['feedback']}`"
                )
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Main header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
# 🔍 PlagIntel
### AI-Driven Plagiarism Intelligence using IBM Granite
""")


# ══════════════════════════════════════════════════════════════════════════════
# Three top-level tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_analyse, tab_dashboard, tab_analytics = st.tabs([
    "🔬 Analyse Submission",
    "📊 Dashboard",
    "📈 Analytics",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB: Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    render_dashboard()


# ══════════════════════════════════════════════════════════════════════════════
# TAB: Analytics
# ══════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    render_analytics()


# ══════════════════════════════════════════════════════════════════════════════
# TAB: Analyse Submission
# ══════════════════════════════════════════════════════════════════════════════

with tab_analyse:

    # ── Agent progress tracker ────────────────────────────────────────────────
    STEPS = [
        "1. Accept",
        "2. Compare",
        "3. Risk Score",
        "4. Granite AI",
        "5. Feedback",
        "6. Save & Learn",
    ]

    def render_progress(current_step: int):
        """Render a compact IBM-styled step progress bar."""
        cols = st.columns(len(STEPS))
        for i, (col, label) in enumerate(zip(cols, STEPS)):
            step_num = i + 1
            if step_num < current_step:
                col.success(label)
            elif step_num == current_step:
                col.info(f"**{label}**")
            else:
                col.markdown(
                    f'<div style="border:1px solid #e0e0e0;border-radius:4px;'
                    f'padding:6px 8px;font-size:12px;color:#525252;text-align:center;">'
                    f'{label}</div>',
                    unsafe_allow_html=True,
                )

    render_progress(st.session_state["step"])
    st.divider()


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Accept Assignment
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown(
        '<h3 style="color:#0f62fe;">Step 1 · Accept Assignment</h3>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Student Information**")
        student_name  = st.text_input("Student Name",     placeholder="e.g. John Doe")
        student_id    = st.text_input("Student ID",       placeholder="e.g. S011")
        assign_title  = st.text_input("Assignment Title", placeholder="e.g. Essay on Climate Change")

    with col_right:
        st.markdown("**Assignment Text**")
        input_method = st.radio(
            "Input method",
            ["Paste text", "Upload file (TXT, PDF, DOCX)"],
            horizontal=True,
        )

        assignment_text = ""

        if input_method == "Paste text":
            assignment_text = st.text_area(
                "Paste the assignment here",
                height=220,
                placeholder="Enter the student's full assignment text…",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a TXT, PDF, or DOCX file",
                type=["txt", "pdf", "docx"],
            )
            if uploaded_file is not None:
                assignment_text, extract_error = extract_text_from_file(uploaded_file)
                if extract_error:
                    st.error(extract_error)
                elif assignment_text.strip():
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
                    with st.expander("Preview extracted text"):
                        st.write(
                            assignment_text[:800]
                            + ("…" if len(assignment_text) > 800 else "")
                        )
                else:
                    st.warning("No text could be extracted. Please check the file.")

    # ── Analyse button ────────────────────────────────────────────────────────

    analyse_clicked = st.button(
        "🚀 Analyse Submission",
        type="primary",
        disabled=(not assignment_text.strip()),
    )

    if analyse_clicked and assignment_text.strip():
        if not student_name.strip():
            st.warning("Please enter the student's name before analysing.")
            st.stop()

        # Capture student info + submission timestamp into session state
        st.session_state["s_student_id"]    = student_id.strip()
        st.session_state["s_student_name"]  = student_name.strip()
        st.session_state["s_assign_title"]  = assign_title.strip()
        st.session_state["s_submit_time"]   = datetime.now().strftime("%Y-%m-%d %H:%M")

        st.session_state["new_text"]        = assignment_text.strip()
        st.session_state["results"]         = None
        st.session_state["overall_risk"]    = None
        st.session_state["granite_explain"] = None
        st.session_state["hybrid_result"]   = None
        st.session_state["feedback_saved"]  = False
        st.session_state["step"]            = 2

        # ── Step 2: Compare (TF-IDF baseline) ────────────────────────────────
        with st.spinner("🔄 Step 2: Comparing with stored submissions…"):
            results = compare_with_stored(assignment_text.strip())
            st.session_state["results"] = results

        # ── Step 3: Risk + hybrid analysis ───────────────────────────────────
        overall = calculate_overall_risk(results)
        st.session_state["overall_risk"] = overall
        st.session_state["step"] = 3

        if overall["top_match"]:
            with st.spinner("🔬 Step 3: Running hybrid detection engine…"):
                hybrid = run_hybrid_analysis(
                    new_text   = assignment_text.strip(),
                    top_match  = overall["top_match"],
                    student_id = student_id.strip(),
                )
                st.session_state["hybrid_result"] = hybrid
        else:
            st.session_state["hybrid_result"] = None

        # ── Step 4: Granite + learning context + hybrid scores ───────────────
        st.session_state["step"] = 4
        if overall["top_match"]:
            learning_ctx = get_learning_context()
            hybrid       = st.session_state["hybrid_result"]
            with st.spinner("🤖 Step 4: Asking IBM Granite to analyse the findings…"):
                granite_result = explain_with_granite(
                    new_text         = assignment_text.strip(),
                    matched_text     = overall["top_match"]["matched_text"],
                    similarity_score = overall["top_match"]["similarity_score"],
                    risk_level       = overall["top_match"]["risk_level"],
                    learning_context = learning_ctx,
                    hybrid_scores    = hybrid,
                )
                st.session_state["granite_explain"] = granite_result
        else:
            st.session_state["granite_explain"] = first_submission_result()

        st.session_state["step"] = 5
        st.rerun()


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 & 3 — Comparison Results & Risk Score
    # ══════════════════════════════════════════════════════════════════════════

    if st.session_state["results"] is not None:

        results  = st.session_state["results"]
        overall  = st.session_state["overall_risk"]
        new_text = st.session_state["new_text"]

        st.divider()
        st.markdown(
            '<h3 style="color:#0f62fe;">Step 2 · Comparison Results</h3>',
            unsafe_allow_html=True,
        )

        if not results:
            st.info("No previous submissions in the database to compare against.")
        else:
            # ── Top-5 ranked match cards ──────────────────────────────────────
            st.markdown("#### 🏆 Top 5 Most Similar Submissions")
            top5      = overall.get("top_matches", results[:5])
            card_cols = st.columns(min(len(top5), 5))
            for i, (col, match) in enumerate(zip(card_cols, top5)):
                pct    = match["similarity_score"] * 100
                risk   = match["risk_level"]
                colour = (
                    "#da1e28" if "High"   in risk else
                    "#ff832b" if "Medium" in risk else
                    "#198038"
                )
                short_title = (
                    match["assignment_title"][:28] + "…"
                    if len(match["assignment_title"]) > 28
                    else match["assignment_title"]
                )
                col.markdown(
                    f"""
                    <div style="background:#ffffff;border:1px solid #e0e0e0;
                                border-top:4px solid {colour};border-radius:6px;
                                padding:14px 10px;text-align:center;
                                box-shadow:0 1px 4px rgba(0,0,0,.06);">
                        <div style="font-size:10px;color:#525252;font-weight:600;
                                    text-transform:uppercase;letter-spacing:.5px;">
                            #{i+1}</div>
                        <div style="font-weight:700;font-size:13px;margin:6px 0 4px;
                                    color:#161616;">{match['student_name']}</div>
                        <div style="font-size:26px;font-weight:800;color:{colour};
                                    line-height:1.1;">{pct:.1f}%</div>
                        <div style="font-size:10px;color:#525252;margin-top:4px;">{risk}</div>
                        <div style="font-size:10px;color:#6f6f6f;margin-top:3px;">{short_title}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

            # Full sortable comparison table
            st.markdown("#### 📋 Full Comparison Table")
            display_data = [
                {
                    "Rank":       i + 1,
                    "Student ID": r["student_id"],
                    "Student":    r["student_name"],
                    "Assignment": r["assignment_title"],
                    "Similarity": f"{r['similarity_score'] * 100:.1f}%",
                    "Risk Level": r["risk_level"],
                }
                for i, r in enumerate(results)
            ]
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

        # ── Step 3: Hybrid Risk Score panel ──────────────────────────────────
        st.divider()
        st.markdown(
            '<h3 style="color:#0f62fe;">Step 3 · Plagiarism Risk Score</h3>',
            unsafe_allow_html=True,
        )

        hybrid = st.session_state.get("hybrid_result")

        # ── Row 1: primary metrics ────────────────────────────────────────────
        if hybrid:
            hv = int(hybrid["final_score"] * 100)
            hc = "#da1e28" if hv >= 65 else "#ff832b" if hv >= 35 else "#198038"
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Hybrid Final Score",   f"{hv}%",
                      help="Weighted score: phrases 30% + sentences 25% + semantic 25% + style 10% + feedback 10%")
            m2.metric("TF-IDF Baseline",      f"{overall['max_score']*100:.1f}%",
                      help="Original TF-IDF cosine similarity (vocabulary overlap).")
            m3.metric("Overall Risk",          hybrid["final_risk"],
                      help="Based on the hybrid weighted score.")
            ai_col = "#da1e28" if hybrid["ai_likelihood"] == "High" else \
                     "#ff832b" if hybrid["ai_likelihood"] == "Medium" else "#198038"
            m4.metric("AI-Assisted Likelihood", hybrid["ai_likelihood"],
                      help="Heuristic only — based on style features, not a definitive detection.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Highest Similarity", f"{overall['max_score']*100:.1f}%")
            m2.metric("Average Similarity", f"{overall['avg_score']*100:.1f}%")
            m3.metric("Overall Risk Level",  overall["risk_level"])

        # ── Row 2: sub-score breakdown ────────────────────────────────────────
        if hybrid:
            st.markdown("<div style='margin:10px 0 4px;font-size:13px;font-weight:600;"
                        "color:#525252;'>Sub-Score Breakdown</div>",
                        unsafe_allow_html=True)
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Exact Phrases",    f"{hybrid['phrase_score']*100:.1f}%",
                      help="30% weight — fraction of submission covered by exact 5+ word phrases.")
            s2.metric("Sentence Sim.",    f"{hybrid['sentence_score']*100:.1f}%",
                      help="25% weight — fraction of sentences with TF-IDF similarity ≥ 0.75.")
            s3.metric("Semantic Sim.",    f"{hybrid['semantic_score']*100:.1f}%",
                      help="25% weight — Lightweight paraphrase similarity (word + char TF-IDF n-grams, pure scikit-learn, CPU-safe).")
            s4.metric("Style Mismatch",   f"{hybrid['style_score']*100:.1f}%",
                      help="10% weight — deviation from this student's past writing style.")
            s5.metric("Feedback Weight",  f"{hybrid['feedback_weight']*100:.1f}%",
                      help="10% weight — adjusted by past teacher decisions at similar scores.")

        # ── Risk badge + bar ──────────────────────────────────────────────────
        if hybrid:
            bar_val = int(hybrid["final_score"] * 100)
            bar_col = "#da1e28" if bar_val >= 65 else "#ff832b" if bar_val >= 35 else "#198038"
            risk_label = hybrid["final_risk"]
        else:
            bar_val = int(overall["max_score"] * 100)
            bar_col = "#da1e28" if bar_val >= 75 else "#ff832b" if bar_val >= 45 else "#198038"
            risk_icons = {"🔴 High Risk": "🔴", "🟡 Medium Risk": "🟡", "🟢 Low Risk": "🟢"}
            risk_label = overall["risk_level"]

        icon = "🔴" if "High" in risk_label else "🟡" if "Medium" in risk_label else "🟢"
        st.markdown(f"### {icon} {risk_label}")

        st.markdown(
            f"""<div style="background:#e0e0e0;border-radius:4px;height:20px;
                            width:100%;margin:8px 0 8px 0;">
                    <div style="background:{bar_col};width:{bar_val}%;height:100%;
                                border-radius:4px;display:flex;align-items:center;
                                padding-left:10px;">
                        <span style="color:white;font-size:12px;font-weight:700;">{bar_val}%</span>
                    </div>
                </div>""",
            unsafe_allow_html=True,
        )

        # ── Style features expander ───────────────────────────────────────────
        if hybrid and hybrid["style_features_new"]:
            with st.expander("✍️ Writing-Style Analysis"):
                sf = hybrid["style_features_new"]
                fa, fb, fc, fd = st.columns(4)
                fa.metric("Avg Sentence Length", f"{sf['avg_sent_len']:.1f} words")
                fb.metric("Vocabulary Richness",  f"{sf['vocab_richness']*100:.1f}%",
                          help="Type-token ratio over first 500 words.")
                fc.metric("Avg Word Length",      f"{sf['avg_word_len']:.1f} chars")
                fd.metric("Formality Score",       f"{sf['formality']*100:.1f}%",
                          help="Fraction of words ≥ 8 characters.")
                if hybrid["style_score"] > 0:
                    st.info(
                        f"**Style mismatch vs past submissions:** {hybrid['style_score']*100:.1f}%  "
                        f"({'Large' if hybrid['style_score']>0.4 else 'Moderate' if hybrid['style_score']>0.2 else 'Small'} shift)"
                    )
                else:
                    st.info("No previous submissions found for this student ID — style comparison not available.")

                # AI-assisted disclaimer
                ai_lk = hybrid["ai_likelihood"]
                ai_colour = "#fef3c7" if ai_lk == "High" else "#f0fdf4"
                ai_border = "#d97706" if ai_lk == "High" else "#16a34a"
                st.markdown(
                    f"""<div style="background:{ai_colour};border-left:3px solid {ai_border};
                            border-radius:4px;padding:10px 14px;font-size:12px;margin-top:8px;">
                        <strong>⚠️ AI-Assisted Writing Likelihood: {ai_lk}</strong><br>
                        This is a heuristic indicator based on writing-style patterns only.
                        It <strong>cannot reliably detect AI-generated text</strong> and should
                        never be used as the sole basis for an academic integrity decision.
                        IBM Granite provides a more contextual assessment in Step 4.
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Matched phrases expander ──────────────────────────────────────────
        if hybrid and hybrid["phrase_list"]:
            with st.expander(f"📌 Exact Matching Phrases ({len(hybrid['phrase_list'])} found)"):
                st.caption("Phrases of 5+ consecutive words found verbatim in both submissions:")
                for i, ph in enumerate(hybrid["phrase_list"][:10], 1):
                    st.markdown(
                        f'<div style="background:#fff7ed;border-left:3px solid #ea580c;'
                        f'border-radius:3px;padding:6px 12px;margin:4px 0;'
                        f'font-size:13px;font-family:monospace;">'
                        f'{i}. "{ph}"</div>',
                        unsafe_allow_html=True,
                    )

        # ── Match highlighting expander ───────────────────────────────────────
        if overall["top_match"]:
            top = overall["top_match"]
            with st.expander("🔆 View Match Highlighting"):
                st.caption(
                    f"Comparing against: **{top['student_name']}** — "
                    f"**{top['assignment_title']}** — "
                    f"Similarity: **{top['similarity_score']*100:.1f}%**"
                )

                # Highlight legend
                st.markdown(
                    """
                    <div style="display:flex;gap:16px;flex-wrap:wrap;
                                font-size:12px;margin-bottom:10px;">
                        <span><mark style="background:#fed7aa;border-radius:2px;
                            border-bottom:2px solid #ea580c;padding:1px 4px;">Exact phrase</mark>
                            &nbsp;3+ consecutive matching words</span>
                        <span><mark style="background:#fef9c3;border-radius:2px;
                            border-bottom:1px dashed #ca8a04;padding:1px 4px;">Similar sentence</mark>
                            &nbsp;High content-word overlap</span>
                        <span><mark style="background:#dbeafe;border-radius:2px;
                            border-bottom:1px dotted #2563eb;padding:1px 4px;">Shared term</mark>
                            &nbsp;Uncommon word in both texts</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                hl_new, hl_matched = highlight_matches(new_text, top["matched_text"])
                h1c, h2c = st.columns(2)
                with h1c:
                    st.markdown("**New Submission**")
                    st.markdown(
                        f'<div style="background:#f4f4f4;border:1px solid #e0e0e0;'
                        f'border-left:3px solid #0f62fe;border-radius:4px;'
                        f'padding:12px 14px;font-size:13px;line-height:1.7;">'
                        f'{hl_new}</div>',
                        unsafe_allow_html=True,
                    )
                with h2c:
                    st.markdown("**Most Similar Stored Submission**")
                    st.markdown(
                        f'<div style="background:#f4f4f4;border:1px solid #e0e0e0;'
                        f'border-left:3px solid #da1e28;border-radius:4px;'
                        f'padding:12px 14px;font-size:13px;line-height:1.7;">'
                        f'{hl_matched}</div>',
                        unsafe_allow_html=True,
                    )

                # TF-IDF baseline note
                st.markdown(
                    """
                    <div style="background:#f0f4ff;border-left:3px solid #0f62fe;
                                border-radius:4px;padding:10px 14px;
                                font-size:12px;color:#393939;margin-top:10px;">
                        <strong>ℹ️ About this similarity score</strong><br>
                        The similarity score is calculated using <strong>TF-IDF + Cosine Similarity</strong>,
                        a baseline method that measures vocabulary overlap between texts.
                        It may flag submissions that share common topic-specific words without any
                        actual plagiarism. Only <strong>exact phrases</strong>, <strong>similar
                        sentences</strong>, and <strong>uncommon shared terms</strong> are highlighted
                        above — generic academic words are excluded.<br><br>
                        <strong>IBM Granite</strong> (Step 4) provides a deeper contextual explanation
                        of whether the overlap is likely copying, paraphrasing, or common knowledge.
                        Always review the Granite analysis before making a final decision.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — IBM Granite Explanation
    # ══════════════════════════════════════════════════════════════════════════

    if st.session_state["granite_explain"] is not None:

        overall        = st.session_state["overall_risk"]
        granite_result = st.session_state["granite_explain"]  # now a dict

        st.divider()
        st.markdown(
            '<h3 style="color:#0f62fe;">Step 4 · IBM Granite Explanation</h3>',
            unsafe_allow_html=True,
        )

        st.markdown("### 🤖 IBM Granite Analysis")
        st.markdown(granite_result["text"])

        # ── IBM Granite Token Usage ───────────────────────────────────────────
        st.markdown(
            '<div style="font-size:13px;font-weight:700;color:#0f62fe;'
            'text-transform:uppercase;letter-spacing:.5px;margin:14px 0 6px;">'
            '🔢 IBM Granite Token Usage</div>',
            unsafe_allow_html=True,
        )
        if granite_result["total_tokens"] == 0:
            st.info(
                "Token usage is unavailable because IBM Granite is not currently "
                "connected. Showing offline analysis above."
            )
        else:
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("Input Tokens",  granite_result["input_tokens"])
            tc2.metric("Output Tokens", granite_result["output_tokens"])
            tc3.metric("Total Tokens",  granite_result["total_tokens"])
            tc4.metric("Model",         granite_result["model_id"])

        # ── Multi-format download (TXT / CSV / PDF) ───────────────────────────
        st.markdown("#### 📥 Download Report")
        top_matches  = overall.get("top_matches", []) if overall else []
        s_name       = st.session_state.get("s_student_name", "Student")
        s_id         = st.session_state.get("s_student_id", "")
        s_title      = st.session_state.get("s_assign_title", "")
        s_risk       = overall["risk_level"]  if overall else ""
        s_score      = overall["max_score"]   if overall else 0.0
        s_time       = st.session_state.get("s_submit_time", "")
        granite_text = granite_result["text"]
        token_usage  = {
            "input_tokens":  granite_result["input_tokens"],
            "output_tokens": granite_result["output_tokens"],
            "total_tokens":  granite_result["total_tokens"],
            "model_id":      granite_result["model_id"],
        }

        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            txt_report = generate_txt_report(
                student_id=s_id, student_name=s_name, assignment_title=s_title,
                similarity_score=s_score, risk_level=s_risk,
                top_matches=top_matches, granite_summary=granite_text,
                submission_time=s_time, token_usage=token_usage,
            )
            st.download_button(
                "📄 TXT Report",
                txt_report,
                file_name=safe_filename(s_name, "txt"),
                mime="text/plain",
            )

        with dl2:
            csv_report = generate_csv_report(
                student_id=s_id, student_name=s_name, assignment_title=s_title,
                similarity_score=s_score, risk_level=s_risk,
                top_matches=top_matches, granite_summary=granite_text,
                submission_time=s_time, token_usage=token_usage,
            )
            st.download_button(
                "📊 CSV Report",
                csv_report,
                file_name=safe_filename(s_name, "csv"),
                mime="text/csv",
            )

        with dl3:
            try:
                pdf_bytes = generate_pdf_report(
                    student_id=s_id, student_name=s_name, assignment_title=s_title,
                    similarity_score=s_score, risk_level=s_risk,
                    top_matches=top_matches, granite_summary=granite_text,
                    submission_time=s_time, token_usage=token_usage,
                )
                st.download_button(
                    "📑 PDF Report",
                    pdf_bytes,
                    file_name=safe_filename(s_name, "pdf"),
                    mime="application/pdf",
                )
            except Exception as pdf_err:
                st.caption(f"PDF unavailable: {pdf_err}")


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Teacher Feedback
    # ══════════════════════════════════════════════════════════════════════════

    if st.session_state["step"] >= 5 and st.session_state["granite_explain"] is not None:

        st.divider()
        st.markdown(
            '<h3 style="color:#0f62fe;">Step 5 · Teacher Feedback</h3>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "Review the analysis above and record your professional judgement. "
            "Your decision is **permanently saved** and will improve future Granite analyses."
        )

        feedback_options = {
            "✅ Genuine — original work, no plagiarism":     "Genuine",
            "⚠️  Suspicious — needs further investigation":  "Suspicious",
            "🚨 Confirmed Plagiarism — action required":     "Confirmed Plagiarism",
            "❌  False Positive — system was wrong":         "False Positive",
        }

        selected_label = st.radio("Your decision:", list(feedback_options.keys()), index=0)
        teacher_notes  = st.text_area(
            "Additional notes (optional)", placeholder="Add any notes…", height=90
        )

        # ── Step 6: Save ──────────────────────────────────────────────────────
        save_clicked = st.button("💾 Save Feedback & Close", type="primary")

        if save_clicked and not st.session_state["feedback_saved"]:
            feedback_value = feedback_options[selected_label]

            overall        = st.session_state["overall_risk"]
            granite_result = st.session_state["granite_explain"] or {}
            granite_text   = granite_result.get("text", "") if isinstance(granite_result, dict) else str(granite_result)
            s_id     = st.session_state["s_student_id"]
            s_name   = st.session_state["s_student_name"]
            s_title  = st.session_state["s_assign_title"]
            s_time   = st.session_state["s_submit_time"]
            s_score  = overall["max_score"]  if overall else 0.0
            s_risk   = overall["risk_level"] if overall else ""

            # Top match info for history record
            top_m         = overall.get("top_match") if overall else None
            top_m_student = top_m["student_name"]    if top_m else ""
            top_m_assign  = top_m["assignment_title"] if top_m else ""

            # Token usage from Granite result dict
            g_in  = granite_result.get("input_tokens",  0) if isinstance(granite_result, dict) else 0
            g_out = granite_result.get("output_tokens", 0) if isinstance(granite_result, dict) else 0
            g_tot = granite_result.get("total_tokens",  0) if isinstance(granite_result, dict) else 0
            g_mid = granite_result.get("model_id",      "") if isinstance(granite_result, dict) else ""

            # 1. Persist submission to old_submissions.csv (extended schema)
            if s_name:
                add_new_submission(
                    student_id       = s_id   or f"AUTO_{datetime.now().strftime('%H%M%S')}",
                    student_name     = s_name or "Unknown Student",
                    assignment_title = s_title or "Unknown Assignment",
                    submission_text  = st.session_state["new_text"],
                    similarity_score = s_score,
                    risk_level       = s_risk,
                    teacher_feedback = feedback_value,
                    teacher_notes    = teacher_notes.strip(),
                    granite_summary  = granite_text,
                )

            # 2. Permanently record in analysis_history.csv (with top-match + token fields)
            record_analysis(
                student_id               = s_id,
                student_name             = s_name or "Unknown",
                assignment_title         = s_title or "Unknown",
                similarity_score         = s_score,
                risk_level               = s_risk,
                teacher_decision         = feedback_value,
                teacher_notes            = teacher_notes.strip(),
                granite_summary          = granite_text,
                top_matching_student     = top_m_student,
                top_matching_assignment  = top_m_assign,
                input_tokens             = g_in,
                output_tokens            = g_out,
                total_tokens             = g_tot,
                granite_model_id         = g_mid,
                analysis_timestamp       = s_time,
            )

            # 3. Update in-session feedback log
            st.session_state["feedback_log"].append({
                "timestamp":    s_time,
                "student_name": s_name or "Unknown",
                "risk_level":   s_risk,
                "max_score":    f"{s_score*100:.1f}%",
                "feedback":     feedback_value,
                "notes":        teacher_notes.strip(),
            })

            st.session_state["feedback_saved"] = True
            st.session_state["step"] = 6
            st.rerun()


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Confirmation & Final Report
    # ══════════════════════════════════════════════════════════════════════════

    if st.session_state["step"] == 6 and st.session_state["feedback_saved"]:

        st.divider()
        st.markdown(
            '<h3 style="color:#198038;">Step 6 · Saved ✅</h3>',
            unsafe_allow_html=True,
        )

        last = st.session_state["feedback_log"][-1]

        st.success(
            f"Feedback recorded: **{last['feedback']}** for "
            f"**{last['student_name']}** at {last['timestamp']}."
        )

        if last["notes"]:
            st.info(f"📝 Notes: {last['notes']}")

        st.markdown(
            "The submission has been saved to **old_submissions.csv** and the full analysis "
            "record has been appended to **analysis_history.csv**.  \n"
            "Future comparisons will include this submission. The Dashboard and Analytics "
            "pages have been updated."
        )

        # Final complete report download with teacher decision included
        overall        = st.session_state["overall_risk"]
        granite_result = st.session_state["granite_explain"] or {}
        granite_text   = granite_result.get("text", "") if isinstance(granite_result, dict) else str(granite_result)
        token_usage    = {
            "input_tokens":  granite_result.get("input_tokens",  0) if isinstance(granite_result, dict) else 0,
            "output_tokens": granite_result.get("output_tokens", 0) if isinstance(granite_result, dict) else 0,
            "total_tokens":  granite_result.get("total_tokens",  0) if isinstance(granite_result, dict) else 0,
            "model_id":      granite_result.get("model_id",      "") if isinstance(granite_result, dict) else "",
        }
        top_matches = overall.get("top_matches", []) if overall else []
        s_name      = last["student_name"]
        s_time      = st.session_state.get("s_submit_time", "")

        final_txt = generate_txt_report(
            student_id       = st.session_state.get("s_student_id", ""),
            student_name     = s_name,
            assignment_title = st.session_state.get("s_assign_title", ""),
            similarity_score = overall["max_score"] if overall else 0.0,
            risk_level       = last["risk_level"],
            top_matches      = top_matches,
            granite_summary  = granite_text,
            teacher_decision = last["feedback"],
            teacher_notes    = last["notes"],
            submission_time  = s_time,
            token_usage      = token_usage,
        )

        fc1, fc2 = st.columns(2)
        with fc1:
            st.download_button(
                "📥 Download Final TXT Report",
                final_txt,
                file_name=safe_filename(s_name, "txt"),
                mime="text/plain",
            )
        with fc2:
            final_csv = generate_csv_report(
                student_id       = st.session_state.get("s_student_id", ""),
                student_name     = s_name,
                assignment_title = st.session_state.get("s_assign_title", ""),
                similarity_score = overall["max_score"] if overall else 0.0,
                risk_level       = last["risk_level"],
                top_matches      = top_matches,
                granite_summary  = granite_text,
                teacher_decision = last["feedback"],
                teacher_notes    = last["notes"],
                submission_time  = s_time,
                token_usage      = token_usage,
            )
            st.download_button(
                "📊 Download Final CSV Report",
                final_csv,
                file_name=safe_filename(s_name, "csv"),
                mime="text/csv",
            )

        if st.button("🔄 Analyse Another Submission", type="primary"):
            for key in ["results", "overall_risk", "granite_explain",
                        "new_text", "feedback_saved", "hybrid_result",
                        "s_student_id", "s_student_name",
                        "s_assign_title", "s_submit_time"]:
                default = (
                    False if key == "feedback_saved" else
                    ""    if key.startswith("s_") or key == "new_text" else
                    None
                )
                st.session_state[key] = default
            st.session_state["step"] = 1
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    '<div style="text-align:center;font-size:11px;color:#525252;">'
    'PlagIntel · Powered by IBM Granite on watsonx.ai · Built with Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
