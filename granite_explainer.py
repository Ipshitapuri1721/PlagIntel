# granite_explainer.py
# Uses IBM Granite via watsonx.ai to explain plagiarism risk.
# ENHANCED: 7-section structured prompt with learning context injection,
# Confidence Score, Potential Cause, and strict grounding rules.

import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

WATSONX_API_KEY  = os.getenv("WATSONX_API_KEY", "")
WATSONX_URL      = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT  = os.getenv("WATSONX_PROJECT_ID", "")

GRANITE_MODEL_ID = "ibm/granite-4-h-small"

GENERATION_PARAMS = {
    "decoding_method":    "greedy",
    "max_new_tokens":     700,   # room for all 7 sections
    "min_new_tokens":     100,
    "repetition_penalty": 1.1,
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    new_text: str,
    matched_text: str,
    similarity_score: float,
    risk_level: str,
    learning_context: str = "",
    hybrid_scores: dict | None = None,   # NEW: sub-scores from run_hybrid_analysis()
) -> str:
    """
    Build a structured, grounded prompt for a 7-section case report.

    Sections:
        ## Case Summary
        ## Reason for Similarity
        ## Matching Concepts
        ## Suspicious Matching Phrases
        ## Potential Cause
        ## Copy Type  (now includes Possible AI-Assisted Writing)
        ## Teacher Recommendation

    Injects optional learning_context from past teacher decisions.
    When hybrid_scores are provided, Granite sees all sub-scores for richer reasoning.
    Strict grounding rules prevent hallucination.
    """
    learning_section = (
        f"\n{learning_context}\n" if learning_context.strip() else ""
    )

    # Build a concise hybrid-score block to include in the prompt when available
    if hybrid_scores:
        sem_line = (
            f"  Semantic similarity    : {hybrid_scores['semantic_score']*100:.1f}%"
            f" (lightweight paraphrase similarity)"
            if hybrid_scores.get("semantic_available")
            else "  Semantic similarity    : N/A"
        )
        hybrid_block = f"""
HYBRID DETECTION SCORES (use as additional context):
  Exact phrase match     : {hybrid_scores['phrase_score']*100:.1f}%  (30% weight)
  Sentence similarity    : {hybrid_scores['sentence_score']*100:.1f}%  (25% weight)
{sem_line}  (25% weight)
  Writing-style mismatch : {hybrid_scores['style_score']*100:.1f}%  (10% weight)
  Feedback weight        : {hybrid_scores['feedback_weight']*100:.1f}%  (10% weight)
  Weighted final score   : {hybrid_scores['final_score']*100:.1f}%
  AI-assisted likelihood : {hybrid_scores['ai_likelihood']} (heuristic only — not definitive)
"""
    else:
        hybrid_block = ""

    prompt = f"""You are a professional academic integrity assistant helping a teacher review student assignments.
{learning_section}{hybrid_block}
STUDENT SUBMISSION:
\"\"\"
{new_text.strip()}
\"\"\"

MATCHED PREVIOUS SUBMISSION:
\"\"\"
{matched_text.strip()}
\"\"\"

TF-IDF Similarity Score : {similarity_score * 100:.1f}%
Risk Level              : {risk_level}

Write a professional plagiarism case report in Markdown using EXACTLY these seven sections, followed by a Confidence Score:

## Case Summary
Write 2–3 sentences: describe the overall risk, what the hybrid scores suggest, and what the teacher should pay attention to.

## Reason for Similarity
Explain, using ONLY the two texts above, why they are similar. Focus on shared vocabulary, sentence patterns, or structural overlap you can actually observe in the texts.

## Matching Concepts
List the main ideas or topics that appear in BOTH texts. Use bullet points. Base this only on the supplied texts.

## Suspicious Matching Phrases
List up to 5 phrases that appear verbatim or near-verbatim in BOTH texts.
Format: `- "exact phrase here"`
If no exact match exists in both texts, write:
> No exact matching phrase found.

## Potential Cause
Choose ONE and briefly explain using only the supplied texts:
- Shared lecture notes or course material
- Direct copy with minor edits
- Independent paraphrasing of the same source
- Common domain vocabulary (not plagiarism)
- Unknown — cannot determine from text alone

## Copy Type
Choose ONE and justify using only the supplied texts:
- **Direct Copy** — large sections reproduced verbatim
- **Heavy Paraphrasing** — ideas and structure reworded but clearly derived
- **Possible AI-Assisted Writing** — unusually formal, highly consistent prose that may have been AI-generated (note: this is a heuristic indicator, not a definitive judgment)
- **Common Knowledge** — similarity reflects standard topic vocabulary, not copying
- **Low Concern** — scores are low and the overlap is explainable

## Teacher Recommendation
One specific, actionable recommendation for the teacher based on the above analysis.

**Confidence Score:** [Low / Medium / High] — briefly explain why.

---

STRICT RULES — MUST FOLLOW:
- Do NOT fabricate or invent any phrases, sentences, or words.
- ONLY quote text that literally appears in the Student Submission or Matched Submission above.
- If you are uncertain about any claim, say so explicitly.
- Base every section strictly on the two supplied texts.
- Do NOT reference any outside knowledge about the topic.
- For AI-Assisted Writing: only select this if the hybrid scores and writing-style features strongly suggest it. Always include a caveat that it is not certain.
"""
    return prompt


# ── Granite call ──────────────────────────────────────────────────────────────

def explain_with_granite(
    new_text: str,
    matched_text: str,
    similarity_score: float,
    risk_level: str,
    learning_context: str = "",
    hybrid_scores: dict | None = None,   # NEW: passed through to build_prompt()
) -> str:
    """
    Call IBM Granite on watsonx.ai and return the structured case report.
    Injects historical learning context and hybrid sub-scores when available.
    Falls back to the offline rule-based explainer when credentials are absent.
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT:
        return _offline_explanation(new_text, matched_text, similarity_score, risk_level,
                                    hybrid_scores=hybrid_scores)

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        model = ModelInference(
            model_id=GRANITE_MODEL_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT,
            params=GENERATION_PARAMS,
        )
        prompt   = build_prompt(new_text, matched_text, similarity_score,
                                risk_level, learning_context, hybrid_scores)
        response = model.generate_text(prompt=prompt)
        return response.strip()

    except Exception as error:
        return (
            f"⚠️ IBM Granite could not be reached.\n\nError: {error}\n\n"
            + _offline_explanation(new_text, matched_text, similarity_score, risk_level,
                                   hybrid_scores=hybrid_scores)
        )


# ── Offline fallback ──────────────────────────────────────────────────────────

def _offline_explanation(
    new_text: str,
    matched_text: str,
    similarity_score: float,
    risk_level: str,
    hybrid_scores: dict | None = None,
) -> str:
    """
    Rule-based 7-section explanation when watsonx.ai is unavailable.
    Mirrors the live prompt structure exactly so the UI renders identically.
    When hybrid_scores are available, the summary and copy-type are refined using them.
    """
    score_pct   = similarity_score * 100
    hybrid_line = ""
    if hybrid_scores:
        fs = hybrid_scores["final_score"] * 100
        ai = hybrid_scores["ai_likelihood"]
        hybrid_line = (
            f"\n\n**Hybrid detection score:** {fs:.1f}%  |  "
            f"**AI-assisted likelihood:** {ai}"
        )

    if similarity_score >= 0.75:
        summary = (
            f"The submission has a **{score_pct:.1f}% similarity score** against a "
            f"stored submission, classified as **{risk_level}**. This indicates a high "
            f"likelihood of copied or heavily paraphrased content requiring immediate review."
        )
        reason = (
            "The two texts share a very large proportion of the same vocabulary and "
            "sentence fragments. The overlap extends beyond common topical terminology "
            "and includes specific phrasings found in both submissions."
        )
        copy_type = (
            "**Direct Copy** — The high similarity and overlapping specific phrases suggest "
            "significant portions were reproduced verbatim or with minimal alterations."
        )
        potential_cause = (
            "Direct copy with minor edits — the extent of overlap makes independent "
            "authorship unlikely without attribution."
        )
        confidence = "**Confidence Score:** High — similarity is well above the 75% threshold."
        recommendation = (
            "Ask the student to explain the overlapping content in person. Compare both "
            "texts side-by-side and consider involving the academic integrity committee."
        )
    elif similarity_score >= 0.45:
        summary = (
            f"The submission scores **{score_pct:.1f}% similarity**, classified as "
            f"**{risk_level}**. Several shared phrases and ideas warrant further review."
        )
        reason = (
            "Both texts address the same topic with overlapping vocabulary and similar "
            "sentence structures. This could reflect paraphrasing or use of a common source."
        )
        copy_type = (
            "**Heavy Paraphrasing** — The moderate similarity suggests ideas and structure "
            "may have been reworded from another submission rather than written independently."
        )
        potential_cause = (
            "Independent paraphrasing of the same source — the student may have reworded "
            "the same materials without proper attribution."
        )
        confidence = "**Confidence Score:** Medium — similarity is in the ambiguous range."
        recommendation = (
            "Request the student's source list and compare the texts structurally. "
            "Look for patterns that go beyond common topic vocabulary."
        )
    else:
        summary = (
            f"The submission has a **{score_pct:.1f}% similarity score**, classified as "
            f"**{risk_level}**. The overlap is likely attributable to shared topic vocabulary."
        )
        reason = (
            "The shared terms are typical for this subject area. The overall sentence "
            "structure and wording appear independently composed."
        )
        copy_type = (
            "**Common Knowledge** — The similarity reflects standard terminology for the "
            "topic rather than copying or paraphrasing."
        )
        potential_cause = (
            "Common domain vocabulary — students writing on the same topic will naturally "
            "use similar subject-specific terms."
        )
        confidence = "**Confidence Score:** Low — similarity is below the threshold for concern."
        recommendation = (
            "No immediate action required. Confirm that any shared phrases are standard "
            "domain knowledge before closing the case."
        )

    # Detect overlapping non-trivial words for the phrases section
    stopwords = {
        "the","a","an","is","in","of","to","and","for","on","at","with","by",
        "are","was","were","it","this","that","be","as","from","or","its","also",
        "which","has","have","been","not","but","we","they","their","our","can",
        "may","more","some","most","all","both","into","than","these","those",
        "will","would","could","should","such","about","after","before","other",
    }
    words_a = {w.lower().strip(".,;:!?\"'") for w in new_text.split()}
    words_b = {w.lower().strip(".,;:!?\"'") for w in matched_text.split()}
    shared  = sorted(
        [w for w in (words_a & words_b) if w not in stopwords and len(w) > 5],
        key=len, reverse=True,
    )[:5]
    phrases_md = (
        "\n".join(f'- "{w}"' for w in shared)
        if shared else
        "> No exact matching phrase found."
    )

    # Build matching concepts from shared long words
    concepts = shared[:4] if shared else []
    concepts_md = (
        "\n".join(f"- {c}" for c in concepts)
        if concepts else
        "- No distinct overlapping concepts identified beyond topic vocabulary."
    )

    return (
        f"## Case Summary\n{summary}{hybrid_line}\n\n"
        f"## Reason for Similarity\n{reason}\n\n"
        f"## Matching Concepts\n{concepts_md}\n\n"
        f"## Suspicious Matching Phrases\n{phrases_md}\n\n"
        f"## Potential Cause\n{potential_cause}\n\n"
        f"## Copy Type\n{copy_type}\n\n"
        f"## Teacher Recommendation\n{recommendation}\n\n"
        f"{confidence}\n\n"
        f"---\n"
        f"_⚙️ Offline analysis — add `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` "
        f"to a `.env` file to enable live IBM Granite explanations._"
    )
