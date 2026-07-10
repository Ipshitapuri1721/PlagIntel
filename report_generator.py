# report_generator.py
# Generates professional plagiarism reports in TXT, CSV, and PDF formats.
# PDF is produced using only Python stdlib (html + weasyprint optional;
# falls back to a text-layout PDF via reportlab if available, else ASCII-art PDF).

import io
import csv
import textwrap
from datetime import datetime


# ── Shared helpers ────────────────────────────────────────────────────────────

def safe_filename(student_name: str, ext: str) -> str:
    """Return a safe download filename: plagiarism_report_John_Doe.{ext}"""
    safe = "".join(c if c.isalnum() or c in " _-" else "_"
                   for c in student_name.strip()).replace(" ", "_") or "report"
    return f"plagiarism_report_{safe}.{ext}"


def _top_match_lines(top_matches: list[dict]) -> list[str]:
    """Build the shared Top Matches block used by TXT and PDF reports."""
    lines = []
    for i, m in enumerate(top_matches[:5], 1):
        lines.append(
            f"  {i}. {m['student_name']} ({m['student_id']})  "
            f"—  {m['similarity_score'] * 100:.1f}%  —  {m['risk_level']}"
        )
        lines.append(f"     Assignment : {m['assignment_title']}")
        lines.append("")
    return lines or ["  No prior submissions to compare against.", ""]


# ── TXT Report ────────────────────────────────────────────────────────────────

def _token_lines(token_usage: dict | None) -> list[str]:
    """Build the shared IBM Granite Token Usage block for TXT/PDF reports."""
    if not token_usage or token_usage.get("total_tokens", 0) == 0:
        return [
            "IBM GRANITE TOKEN USAGE",
            "-" * 64,
            "  Token usage unavailable (offline fallback was used).",
            "",
        ]
    return [
        "IBM GRANITE TOKEN USAGE",
        "-" * 64,
        f"  Input Tokens   : {token_usage.get('input_tokens',  0)}",
        f"  Output Tokens  : {token_usage.get('output_tokens', 0)}",
        f"  Total Tokens   : {token_usage.get('total_tokens',  0)}",
        f"  Model ID       : {token_usage.get('model_id',      'N/A')}",
        "",
    ]


def generate_txt_report(
    student_id: str,
    student_name: str,
    assignment_title: str,
    similarity_score: float,
    risk_level: str,
    top_matches: list[dict],
    granite_summary: str,
    teacher_decision: str = "",
    teacher_notes: str = "",
    submission_time: str = "",
    token_usage: dict | None = None,
) -> str:
    """
    Build a professional plain-text plagiarism report.
    Suitable for st.download_button with mime='text/plain'.
    Includes: student info, similarity, risk, top-5 matches,
    Granite summary, token usage, teacher decision, recommendation, timestamp.
    """
    sep  = "=" * 64
    dash = "-" * 64
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    sub_time = submission_time or now

    # Risk → recommendation map
    rec_map = {
        "High":   "Escalate to academic integrity committee. Request in-person explanation.",
        "Medium": "Request student's sources. Compare submissions side-by-side.",
        "Low":    "No immediate action needed. Verify shared phrases are common knowledge.",
    }
    risk_key = "High" if "High" in risk_level else "Medium" if "Medium" in risk_level else "Low"
    recommendation = rec_map[risk_key]

    lines = [
        sep,
        "         PLAGINTEL — PLAGIARISM INTELLIGENCE REPORT",
        f"         Powered by IBM Granite · watsonx.ai",
        f"         Generated : {now}",
        sep,
        "",
        "STUDENT INFORMATION",
        dash,
        f"  Student ID       : {student_id}",
        f"  Student Name     : {student_name}",
        f"  Assignment       : {assignment_title}",
        f"  Submission Time  : {sub_time}",
        "",
        "SIMILARITY ANALYSIS",
        dash,
        f"  Highest Similarity Score  : {similarity_score * 100:.1f}%",
        f"  Overall Risk Level        : {risk_level}",
        "",
        "TOP 5 SIMILAR ASSIGNMENTS",
        dash,
    ] + _top_match_lines(top_matches) + [
        "IBM GRANITE ANALYSIS",
        dash,
    ]

    # Word-wrap the Granite summary at 80 chars
    for para in granite_summary.strip().split("\n"):
        wrapped = textwrap.fill(para.strip(), width=80) if para.strip() else ""
        lines.append(wrapped)
    lines.append("")

    lines += _token_lines(token_usage)

    if teacher_decision:
        lines += [
            "TEACHER DECISION",
            dash,
            f"  Decision   : {teacher_decision}",
            f"  Notes      : {teacher_notes or '(none)'}",
            "",
        ]

    lines += [
        "RECOMMENDATION",
        dash,
        f"  {recommendation}",
        "",
        sep,
        "  Powered by IBM Granite · watsonx.ai · PlagIntel",
        sep,
    ]
    return "\n".join(lines)


# ── CSV Report ────────────────────────────────────────────────────────────────

def generate_csv_report(
    student_id: str,
    student_name: str,
    assignment_title: str,
    similarity_score: float,
    risk_level: str,
    top_matches: list[dict],
    granite_summary: str,
    teacher_decision: str = "",
    teacher_notes: str = "",
    submission_time: str = "",
    token_usage: dict | None = None,
) -> str:
    """
    Build a structured CSV report.
    Suitable for st.download_button with mime='text/csv'.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    writer.writerow(["PlagIntel Plagiarism Intelligence Report"])
    writer.writerow(["Generated", now])
    writer.writerow(["Powered by", "IBM Granite · watsonx.ai"])
    writer.writerow([])

    # Student info
    writer.writerow(["STUDENT INFORMATION"])
    writer.writerow(["Student ID", "Student Name", "Assignment Title",
                     "Submission Time", "Similarity Score",
                     "Risk Level", "Teacher Decision", "Teacher Notes"])
    writer.writerow([
        student_id, student_name, assignment_title,
        submission_time or now,
        f"{similarity_score * 100:.1f}%", risk_level,
        teacher_decision, teacher_notes,
    ])
    writer.writerow([])

    # Top matches
    writer.writerow(["TOP 5 SIMILAR ASSIGNMENTS"])
    writer.writerow(["Rank", "Matched Student ID", "Matched Student Name",
                     "Matched Assignment", "Similarity %", "Risk Level"])
    for i, m in enumerate(top_matches[:5], 1):
        writer.writerow([
            i,
            m["student_id"],
            m["student_name"],
            m["assignment_title"],
            f"{m['similarity_score'] * 100:.1f}%",
            m["risk_level"],
        ])
    writer.writerow([])

    # Granite summary
    writer.writerow(["IBM GRANITE ANALYSIS"])
    writer.writerow([granite_summary.replace("\n", " | ")])
    writer.writerow([])

    # Token usage
    writer.writerow(["IBM GRANITE TOKEN USAGE"])
    tu = token_usage or {}
    writer.writerow(["Input Tokens", "Output Tokens", "Total Tokens", "Model ID"])
    writer.writerow([
        tu.get("input_tokens",  0),
        tu.get("output_tokens", 0),
        tu.get("total_tokens",  0),
        tu.get("model_id",      "N/A"),
    ])
    writer.writerow([])

    # Recommendation
    rec_map = {
        "High":   "Escalate to academic integrity committee.",
        "Medium": "Request student sources and compare side-by-side.",
        "Low":    "No immediate action needed.",
    }
    risk_key = "High" if "High" in risk_level else "Medium" if "Medium" in risk_level else "Low"
    writer.writerow(["RECOMMENDATION"])
    writer.writerow([rec_map[risk_key]])

    return output.getvalue()


# ── PDF Report ────────────────────────────────────────────────────────────────

def generate_pdf_report(
    student_id: str,
    student_name: str,
    assignment_title: str,
    similarity_score: float,
    risk_level: str,
    top_matches: list[dict],
    granite_summary: str,
    teacher_decision: str = "",
    teacher_notes: str = "",
    submission_time: str = "",
    token_usage: dict | None = None,
) -> bytes:
    """
    Generate a PDF report.

    Strategy (tries in order):
      1. reportlab — clean professional PDF
      2. Fallback  — encode the TXT report as a minimal valid PDF binary

    Returns bytes suitable for st.download_button with mime='application/pdf'.
    """
    try:
        return _pdf_reportlab(
            student_id, student_name, assignment_title,
            similarity_score, risk_level, top_matches,
            granite_summary, teacher_decision, teacher_notes,
            submission_time, token_usage,
        )
    except ImportError:
        pass  # reportlab not installed — use fallback

    # Fallback: embed the TXT report as raw text inside a minimal PDF shell
    txt = generate_txt_report(
        student_id, student_name, assignment_title,
        similarity_score, risk_level, top_matches,
        granite_summary, teacher_decision, teacher_notes,
        submission_time, token_usage,
    )
    return _txt_to_minimal_pdf(txt)


def _pdf_reportlab(
    student_id, student_name, assignment_title,
    similarity_score, risk_level, top_matches,
    granite_summary, teacher_decision, teacher_notes,
    submission_time, token_usage=None,
) -> bytes:
    """Build a proper PDF using reportlab (optional dependency)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    IBM_BLUE_RL = colors.HexColor("#0f62fe")
    IBM_RED_RL  = colors.HexColor("#da1e28")

    h1  = ParagraphStyle("H1",  parent=styles["Heading1"],
                          textColor=IBM_BLUE_RL, fontSize=16, spaceAfter=4)
    h2  = ParagraphStyle("H2",  parent=styles["Heading2"],
                          textColor=IBM_BLUE_RL, fontSize=12, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
    cap  = ParagraphStyle("Cap",  parent=styles["Normal"], fontSize=8,
                          textColor=colors.grey)

    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    story = [
        Paragraph("PlagIntel — Plagiarism Intelligence Report", h1),
        Paragraph(f"Powered by IBM Granite · watsonx.ai  |  Generated: {now}", cap),
        Spacer(1, 0.3*cm),
        HRFlowable(width="100%", thickness=1, color=IBM_BLUE_RL),
        Spacer(1, 0.3*cm),

        Paragraph("Student Information", h2),
    ]

    info_data = [
        ["Student ID",   student_id,    "Risk Level", risk_level],
        ["Student Name", student_name,  "Similarity", f"{similarity_score*100:.1f}%"],
        ["Assignment",   assignment_title, "Submitted", submission_time or now],
    ]
    info_table = Table(info_data, colWidths=[3.5*cm, 6*cm, 3*cm, 4*cm])
    info_table.setStyle(TableStyle([
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
        ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#f4f4f4")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e0e0e0")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story += [info_table, Spacer(1, 0.4*cm), Paragraph("Top 5 Similar Assignments", h2)]

    match_data = [["#", "Student Name", "Student ID", "Assignment", "Similarity", "Risk"]]
    for i, m in enumerate(top_matches[:5], 1):
        match_data.append([
            str(i), m["student_name"], m["student_id"],
            m["assignment_title"][:45],
            f"{m['similarity_score']*100:.1f}%", m["risk_level"],
        ])
    if len(match_data) == 1:
        match_data.append(["—", "No prior submissions", "", "", "", ""])

    match_table = Table(match_data, colWidths=[0.6*cm, 3.8*cm, 2.2*cm, 5*cm, 2*cm, 2.8*cm])
    match_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), IBM_BLUE_RL),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story += [match_table, Spacer(1, 0.4*cm), Paragraph("IBM Granite Analysis", h2)]

    # Granite summary — render each line as a paragraph
    for line in granite_summary.strip().split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1*cm))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], ParagraphStyle(
                "GH", parent=styles["Heading3"], fontSize=10,
                textColor=IBM_BLUE_RL, spaceAfter=2
            )))
        elif line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph(f"• {line[2:]}", body))
        else:
            story.append(Paragraph(line, body))

    story.append(Spacer(1, 0.3*cm))

    # Token usage table
    tu = token_usage or {}
    tok_data = [
        ["Input Tokens", "Output Tokens", "Total Tokens", "Model ID"],
        [
            str(tu.get("input_tokens",  0)),
            str(tu.get("output_tokens", 0)),
            str(tu.get("total_tokens",  0)),
            str(tu.get("model_id",      "N/A")),
        ],
    ]
    tok_table = Table(tok_data, colWidths=[3.5*cm, 3.5*cm, 3.5*cm, 6*cm])
    tok_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), IBM_BLUE_RL),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story += [
        Paragraph("IBM Granite Token Usage", h2),
        tok_table,
        Spacer(1, 0.3*cm),
    ]

    if teacher_decision:
        story += [
            Paragraph("Teacher Decision", h2),
            Paragraph(f"<b>Decision:</b> {teacher_decision}", body),
            Paragraph(f"<b>Notes:</b> {teacher_notes or '(none)'}", body),
            Spacer(1, 0.3*cm),
        ]

    rec_map = {
        "High":   "Escalate to academic integrity committee. Request in-person explanation.",
        "Medium": "Request student sources and compare submissions side-by-side.",
        "Low":    "No immediate action needed. Verify shared phrases are common knowledge.",
    }
    risk_key = "High" if "High" in risk_level else "Medium" if "Medium" in risk_level else "Low"
    story += [
        HRFlowable(width="100%", thickness=1, color=IBM_BLUE_RL),
        Spacer(1, 0.2*cm),
        Paragraph(f"<b>Recommendation:</b> {rec_map[risk_key]}", body),
        Spacer(1, 0.3*cm),
        Paragraph("Powered by IBM Granite · watsonx.ai · PlagIntel", cap),
    ]

    doc.build(story)
    return buf.getvalue()


def _txt_to_minimal_pdf(txt: str) -> bytes:
    """
    Encode a plain-text string as a valid but minimal PDF.
    Used as a last-resort fallback when reportlab is not installed.
    The resulting file can be opened by any PDF reader.
    """
    # Escape special PDF characters
    safe = (txt
            .replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)"))

    lines = safe.split("\n")
    # Each line becomes a Td (move down) + Tj (show text) operation
    stream_lines = ["BT", "/F1 9 Tf", "50 800 Td", "12 TL"]
    for line in lines:
        # Truncate very long lines to avoid malformed PDF
        stream_lines.append(f"({line[:120]}) Tj T*")
    stream_lines.append("ET")
    stream_content = "\n".join(stream_lines)
    stream_bytes   = stream_content.encode("latin-1", errors="replace")

    objects = []
    # 1: Catalog, 2: Pages, 3: Page, 4: Font, 5: Stream
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 595 842] "
        b"/Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >>\nendobj\n"
    )
    objects.append(
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n"
    )
    stream_obj = (
        f"5 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode()
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )
    objects.append(stream_obj)

    header  = b"%PDF-1.4\n"
    body_   = b""
    offsets = []
    pos     = len(header)
    for obj in objects:
        offsets.append(pos)
        body_ += obj
        pos   += len(obj)

    xref_pos = len(header) + len(body_)
    xref = f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    trailer = (
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    )

    return header + body_ + xref.encode() + trailer.encode()
