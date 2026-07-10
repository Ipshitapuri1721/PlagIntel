# plagiarism_engine.py
# Handles similarity comparison between a new submission and stored submissions.
# ENHANCED: hybrid detection engine (exact phrases + sentence similarity +
# lightweight paraphrase similarity + writing-style analysis + feedback weighting).
#
# NOTE: Semantic similarity uses a deployment-safe scikit-learn implementation
# (TF-IDF word n-grams + character n-grams + cosine similarity).
# No torch, transformers, sentence-transformers, or CUDA packages are used.
# This is labelled "Lightweight paraphrase similarity" throughout.

import os
import re
import math
import pandas as pd
from collections import Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Constants ─────────────────────────────────────────────────────────────────

SUBMISSIONS_FILE = "data/old_submissions.csv"

HIGH_RISK_THRESHOLD   = 0.75
MEDIUM_RISK_THRESHOLD = 0.45

# All columns in the extended submissions schema
SUBMISSION_COLUMNS = [
    "student_id", "student_name", "assignment_title",
    "submission_text", "submission_date",
    "similarity_score", "risk_level",
    "teacher_feedback", "teacher_notes", "granite_summary",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_submissions() -> pd.DataFrame:
    """
    Load previous submissions from CSV.
    Back-fills any missing extended columns so old CSV files still work.
    """
    os.makedirs("data", exist_ok=True)
    try:
        df = pd.read_csv(SUBMISSIONS_FILE)
        df = df.dropna(subset=["submission_text"])
        # Add any missing columns introduced by the new schema
        for col in SUBMISSION_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)


def save_submissions(df: pd.DataFrame) -> None:
    """Persist the full submissions DataFrame to CSV."""
    os.makedirs("data", exist_ok=True)
    df.to_csv(SUBMISSIONS_FILE, index=False)


def classify_risk(score: float) -> str:
    """Map a 0-1 cosine similarity score to a labelled risk tier."""
    if score >= HIGH_RISK_THRESHOLD:
        return "🔴 High Risk"
    elif score >= MEDIUM_RISK_THRESHOLD:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"


# ── Core engine ───────────────────────────────────────────────────────────────

def compare_with_stored(new_text: str) -> list[dict]:
    """
    Compare *new_text* against every stored submission using TF-IDF cosine similarity.

    Returns a list of result dicts sorted by similarity descending (highest first).
    Each dict contains:
        student_id, student_name, assignment_title,
        similarity_score (0–1), risk_level, matched_text
    """
    df = load_submissions()

    if df.empty:
        return []

    stored_texts = df["submission_text"].tolist()

    # Vectorise all stored texts + new submission together for consistent IDF
    all_texts    = stored_texts + [new_text]
    vectorizer   = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    new_vector    = tfidf_matrix[-1]
    stored_matrix = tfidf_matrix[:-1]

    scores = cosine_similarity(new_vector, stored_matrix).flatten()

    results = []
    for idx, score in enumerate(scores):
        row = df.iloc[idx]
        results.append(
            {
                "student_id":       str(row["student_id"]),
                "student_name":     str(row["student_name"]),
                "assignment_title": str(row["assignment_title"]),
                "similarity_score": round(float(score), 4),
                "risk_level":       classify_risk(score),
                "matched_text":     str(row["submission_text"]),
            }
        )

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results


def get_top_matches(results: list[dict], n: int = 5) -> list[dict]:
    """Return the top-n most similar matches from a compare_with_stored result list."""
    return results[:n]


def calculate_overall_risk(results: list[dict]) -> dict:
    """
    Summarise comparison results into a single risk assessment dict.

    Keys: max_score, avg_score, risk_level, top_match (or None), top_matches (list)
    """
    if not results:
        return {
            "max_score":   0.0,
            "avg_score":   0.0,
            "risk_level":  "🟢 Low Risk",
            "top_match":   None,
            "top_matches": [],
        }

    max_score = results[0]["similarity_score"]
    avg_score = round(sum(r["similarity_score"] for r in results) / len(results), 4)

    return {
        "max_score":   max_score,
        "avg_score":   avg_score,
        "risk_level":  classify_risk(max_score),
        "top_match":   results[0],
        "top_matches": get_top_matches(results, 5),  # NEW: expose top 5
    }


# ── Match highlighting ────────────────────────────────────────────────────────
# Three-tier strategy:
#   Tier 1 — exact phrases of 3+ consecutive words (orange highlight)
#   Tier 2 — sentences with high cosine-like word-overlap (yellow highlight)
#   Tier 3 — uncommon important terms absent from a broad academic stoplist (blue highlight)
#
# Common academic/generic words ("features", "accuracy", "example", "information",
# "process", "method", "system" …) are excluded from Tier 3 so the display is not
# cluttered by vocabulary overlap that has no plagiarism significance.

# Extended stopword set: standard function words + broad academic vocabulary
_STOPWORDS: set[str] = {
    # Function words
    "the","a","an","is","in","of","to","and","for","on","at","with","by",
    "are","was","were","it","this","that","be","as","from","or","its","also",
    "which","has","have","been","not","but","we","they","their","our","can",
    "may","more","some","most","all","both","into","than","these","those",
    "will","would","could","should","such","about","after","before","other",
    "up","out","so","do","did","if","then","when","where","how","what","who",
    "any","each","very","just","now","here","there","while","though","however",
    "thus","hence","therefore","furthermore","moreover","although","because",
    "since","until","unless","even","still","only","first","second","third",
    "one","two","three","many","few","much","well","new","own","same","back",
    "high","low","large","small","long","short","good","best","different",
    # Generic academic / essay words that appear in almost any assignment
    "study","studies","research","result","results","analysis","conclusion",
    "example","examples","approach","method","methods","process","processes",
    "system","systems","model","models","data","information","problem","problems",
    "solution","solutions","feature","features","factor","factors","aspect",
    "aspects","concept","concepts","theory","theories","principle","principles",
    "topic","topics","issue","issues","area","areas","field","fields","work",
    "works","paper","review","type","types","based","used","using","given",
    "number","set","use","form","part","point","level","case","cases","value",
    "values","term","terms","note","notes","effect","effects","impact","impacts",
    "performance","accuracy","quality","measure","measures","evaluation",
    "comparison","application","applications","development","implementation",
    "introduction","objective","objectives","purpose","scope","section",
    "figure","table","chapter","appendix","reference","references","report",
}

# Minimum word length for Tier-3 single-term highlighting
_MIN_TERM_LEN = 7

# Minimum number of unique content words that must overlap in a sentence
# for it to be highlighted as a high-similarity sentence (Tier 2)
_SENTENCE_OVERLAP_THRESHOLD = 3


def _clean_word(w: str) -> str:
    """Strip punctuation from both ends of a word token."""
    return w.strip(".,;:!?\"'()[]{}").lower()


def _content_words(text: str) -> list[str]:
    """Return lower-cased content words (not in _STOPWORDS, length >= _MIN_TERM_LEN)."""
    return [
        _clean_word(w) for w in text.split()
        if _clean_word(w) not in _STOPWORDS
        and len(_clean_word(w)) >= _MIN_TERM_LEN
    ]


def _tokenise_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


# ── Tier 1: exact 3+ word phrases ────────────────────────────────────────────

def _find_exact_phrases(text_a: str, text_b: str, min_words: int = 3) -> list[str]:
    """
    Return all phrases of *min_words* or more consecutive words that appear
    verbatim (case-insensitive) in both texts.

    Only phrases whose words are mostly content words are returned, to avoid
    flagging runs of common function words like "in the case of".
    """
    tokens_a = text_a.lower().split()
    tokens_b = set()
    # Build a set of all n-gram strings from text_b for fast lookup
    tokens_b_list = text_b.lower().split()
    for n in range(min_words, min(12, len(tokens_b_list)) + 1):
        for i in range(len(tokens_b_list) - n + 1):
            tokens_b.add(" ".join(tokens_b_list[i:i+n]))

    found: list[str] = []
    # Slide a window over text_a, checking progressively longer phrases
    i = 0
    while i < len(tokens_a):
        best = None
        # Try longest first so we capture the widest matching phrase
        for n in range(min(12, len(tokens_a) - i), min_words - 1, -1):
            candidate = " ".join(tokens_a[i:i+n])
            if candidate in tokens_b:
                # Require at least 1 content word in the phrase (no pure stopword runs)
                phrase_content = [
                    w for w in candidate.split()
                    if _clean_word(w) not in _STOPWORDS and len(_clean_word(w)) >= 4
                ]
                if phrase_content:
                    best = candidate
                    break
        if best:
            found.append(best)
            i += len(best.split())   # skip past the matched phrase
        else:
            i += 1

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# ── Tier 2: high-similarity sentences ────────────────────────────────────────

def _find_similar_sentences(
    text_a: str, text_b: str, threshold: int = _SENTENCE_OVERLAP_THRESHOLD
) -> tuple[set[str], set[str]]:
    """
    Return (sentences_from_a, sentences_from_b) whose content-word overlap
    is >= *threshold* unique terms.

    Only flags sentences that share meaningfully specific vocabulary — not those
    that share only generic academic words already in _STOPWORDS.
    """
    sents_a = _tokenise_sentences(text_a)
    sents_b = _tokenise_sentences(text_b)

    flagged_a: set[str] = set()
    flagged_b: set[str] = set()

    for sa in sents_a:
        cw_a = set(_content_words(sa))
        if not cw_a:
            continue
        for sb in sents_b:
            cw_b = set(_content_words(sb))
            overlap = cw_a & cw_b
            if len(overlap) >= threshold:
                flagged_a.add(sa)
                flagged_b.add(sb)

    return flagged_a, flagged_b


# ── Tier 3: uncommon important terms ─────────────────────────────────────────

def _find_uncommon_shared_terms(text_a: str, text_b: str) -> set[str]:
    """
    Return terms that appear in both texts AND are not in _STOPWORDS.

    Additionally filter out words that are very frequent in either text
    (frequency > 2 suggests a generic repeated term for this specific document,
    not a unique identifying term worth flagging).
    """
    from collections import Counter

    cw_a = _content_words(text_a)
    cw_b = _content_words(text_b)

    freq_a = Counter(cw_a)
    freq_b = Counter(cw_b)

    shared = set(cw_a) & set(cw_b)

    # Keep only terms that are not overly common in either text
    return {
        w for w in shared
        if freq_a[w] <= 2 and freq_b[w] <= 2
        and len(w) >= _MIN_TERM_LEN
    }


# ── Main highlight function ───────────────────────────────────────────────────

def highlight_matches(new_text: str, matched_text: str) -> tuple[str, str]:
    """
    Return (highlighted_new, highlighted_matched) as HTML strings.

    Uses three tiers of matching quality:
      🟠 Orange  — exact 3+ word phrases (strongest signal)
      🟡 Yellow  — high-similarity sentences (moderate signal)
      🔵 Blue    — uncommon shared terms (supplementary signal)

    Generic academic vocabulary and stopwords are never highlighted alone,
    preventing false-positive noise from common topic words.
    """
    # ── Tier 1: exact phrases ─────────────────────────────────────────────────
    exact_phrases = _find_exact_phrases(new_text, matched_text, min_words=3)

    # ── Tier 2: similar sentences ─────────────────────────────────────────────
    sim_sents_new, sim_sents_matched = _find_similar_sentences(new_text, matched_text)

    # ── Tier 3: uncommon shared terms ────────────────────────────────────────
    uncommon_terms = _find_uncommon_shared_terms(new_text, matched_text)

    # Remove any terms that are already fully covered by an exact phrase
    # (avoid double-highlighting the same text span)
    phrase_words: set[str] = set()
    for ph in exact_phrases:
        phrase_words.update(ph.split())
    uncommon_terms -= phrase_words

    def _apply_highlights(text: str, is_new: bool) -> str:
        """
        Apply all three tiers of highlighting to *text*.
        Order: phrases first (widest spans), then sentences, then single terms.
        """
        result = text

        # Tier 1 — orange phrase highlights
        for phrase in sorted(exact_phrases, key=len, reverse=True):
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            result  = pattern.sub(
                lambda m: (
                    '<mark style="background:#fed7aa;border-radius:2px;'
                    'border-bottom:2px solid #ea580c;" '
                    'title="Exact matching phrase">'
                    f'{m.group()}</mark>'
                ),
                result,
            )

        # Tier 2 — yellow sentence highlights
        sim_sents = sim_sents_new if is_new else sim_sents_matched
        for sent in sim_sents:
            # Only mark the sentence if it hasn't already been fully phrase-highlighted
            escaped = re.escape(sent)
            pattern = re.compile(escaped)
            result  = pattern.sub(
                lambda m: (
                    '<mark style="background:#fef9c3;border-radius:2px;'
                    'border-bottom:1px dashed #ca8a04;" '
                    'title="High-similarity sentence">'
                    f'{m.group()}</mark>'
                ),
                result,
            )

        # Tier 3 — blue uncommon-term highlights (only if not inside an existing <mark>)
        for term in sorted(uncommon_terms, key=len, reverse=True):
            # Skip if this term is a substring of an already-highlighted phrase
            if any(term in ph for ph in exact_phrases):
                continue
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            result  = pattern.sub(
                lambda m: (
                    # Don't re-highlight if already inside a <mark> span
                    m.group()
                    if "<mark" in result[max(0, m.start()-20):m.start()]
                    else (
                        '<mark style="background:#dbeafe;border-radius:2px;'
                        'border-bottom:1px dotted #2563eb;" '
                        'title="Uncommon shared term">'
                        f'{m.group()}</mark>'
                    )
                ),
                result,
            )

        return result

    hl_new     = _apply_highlights(new_text,     is_new=True)
    hl_matched = _apply_highlights(matched_text, is_new=False)
    return hl_new, hl_matched


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
#
# Weighted final score:
#   Exact phrase match      : 30 %
#   Sentence similarity     : 25 %
#   Semantic similarity     : 25 %
#   Writing-style mismatch  : 10 %
#   Teacher feedback weight : 10 %
#
# Each sub-score is 0.0 – 1.0.  The combined score is returned alongside
# individual component scores and a human-readable label for each.

_PHRASE_WEIGHT   = 0.30
_SENTENCE_WEIGHT = 0.25
_SEMANTIC_WEIGHT = 0.25
_STYLE_WEIGHT    = 0.10
_FEEDBACK_WEIGHT = 0.10


# ── Sentence-level similarity (TF-IDF cosine per sentence pair) ───────────────

def _sentence_similarity_score(text_a: str, text_b: str,
                                threshold: float = 0.75) -> float:
    """
    Split both texts into sentences, compare every pair with TF-IDF cosine
    similarity, and return the fraction of sentence pairs that exceed
    *threshold*.  Returns 0.0 when either text has fewer than 2 sentences.
    """
    sents_a = _tokenise_sentences(text_a)
    sents_b = _tokenise_sentences(text_b)

    if len(sents_a) < 2 or len(sents_b) < 2:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sents_a + sents_b)
    except ValueError:
        return 0.0

    mat_a = matrix[: len(sents_a)]
    mat_b = matrix[len(sents_a):]
    sims  = cosine_similarity(mat_a, mat_b)   # shape (|sents_a|, |sents_b|)

    # For each sentence in A, find its best match in B
    best_scores = sims.max(axis=1)
    flagged     = (best_scores >= threshold).sum()
    return round(flagged / len(sents_a), 4)


# ── Lightweight paraphrase similarity (deployment-safe, CPU-only) ─────────────
#
# This is a "deployment-safe semantic approximation" using only scikit-learn.
# It does NOT use torch, transformers, sentence-transformers, or any GPU/CUDA
# packages. It captures paraphrase and morphological similarity by combining:
#   • Word-level TF-IDF n-grams (1, 2) — 60% weight
#   • Character-level TF-IDF n-grams (3, 5) — 40% weight
#   → Cosine similarity of the resulting vectors
#
# Label: "Lightweight paraphrase similarity"

def _semantic_similarity_score(text_a: str, text_b: str) -> float:
    """
    Compute lightweight paraphrase similarity between *text_a* and *text_b*.

    Uses a deployment-safe semantic approximation — pure scikit-learn only:
      - Word TF-IDF n-grams (1, 2) with cosine similarity  → 60% weight
      - Character TF-IDF n-grams (3, 5) with cosine similarity → 40% weight

    No torch, transformers, sentence-transformers, scipy, or CUDA required.
    Runs on any CPU environment including Streamlit Community Cloud.
    Returns 0.0 on any error instead of crashing.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    try:
        word_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
        )

        word_matrix = word_vectorizer.fit_transform([text_a, text_b])
        char_matrix = char_vectorizer.fit_transform([text_a, text_b])

        word_score = cosine_similarity(
            word_matrix[0:1], word_matrix[1:2]
        )[0][0]

        char_score = cosine_similarity(
            char_matrix[0:1], char_matrix[1:2]
        )[0][0]

        return round(float((0.6 * word_score) + (0.4 * char_score)), 4)

    except Exception:
        return 0.0


# ── Exact-phrase score (0-1, ratio of matched phrase words to total words) ────

def _phrase_match_score(text_a: str, text_b: str) -> tuple[float, list[str]]:
    """
    Return (score, phrases_found).
    Score = fraction of words in text_a covered by exact matching phrases (≥5 words).
    Phrases must contain at least 1 non-stopword content word of length ≥ 4.
    """
    phrases = _find_exact_phrases(text_a, text_b, min_words=5)
    if not phrases:
        return 0.0, []
    total_words    = max(len(text_a.split()), 1)
    covered_words  = sum(len(p.split()) for p in phrases)
    score          = round(min(covered_words / total_words, 1.0), 4)
    return score, phrases


# ── Writing-style analysis ────────────────────────────────────────────────────

def _style_features(text: str) -> dict:
    """
    Compute a lightweight style fingerprint for *text*.
    Returns a dict with:
        avg_sent_len    — average sentence length in words
        vocab_richness  — type-token ratio (unique / total words, capped at 500)
        avg_word_len    — average word length in characters
        formality_proxy — ratio of long words (>=8 chars) to total words
    """
    sentences = _tokenise_sentences(text)
    words     = [_clean_word(w) for w in text.split() if _clean_word(w)]

    avg_sent_len   = (sum(len(s.split()) for s in sentences) / max(len(sentences), 1))
    sample         = words[:500]
    vocab_richness = len(set(sample)) / max(len(sample), 1)
    avg_word_len   = sum(len(w) for w in words) / max(len(words), 1)
    formality      = sum(1 for w in words if len(w) >= 8) / max(len(words), 1)

    return {
        "avg_sent_len":   round(avg_sent_len,   2),
        "vocab_richness": round(vocab_richness,  4),
        "avg_word_len":   round(avg_word_len,    2),
        "formality":      round(formality,        4),
    }


def _style_mismatch_score(new_text: str, student_id: str) -> float:
    """
    Compare *new_text* style fingerprint against all previous submissions by
    the same *student_id*.  Returns a mismatch score 0.0 – 1.0.
    0.0 = consistent with past writing.
    1.0 = completely inconsistent (possible ghost-writing or AI substitution).

    Falls back to 0.0 when there are no previous submissions for this student.
    """
    df = load_submissions()
    prior = df[df["student_id"].astype(str) == str(student_id)]["submission_text"].tolist()
    if not prior:
        return 0.0

    new_feats = _style_features(new_text)

    # Average style features across prior submissions
    prior_feats_list = [_style_features(t) for t in prior]
    avg_feats = {
        k: sum(f[k] for f in prior_feats_list) / len(prior_feats_list)
        for k in new_feats
    }

    # Normalised absolute deviation across all 4 features
    deviations = []
    for k in new_feats:
        baseline = avg_feats[k]
        if baseline == 0:
            continue
        deviations.append(abs(new_feats[k] - baseline) / baseline)

    return round(min(sum(deviations) / max(len(deviations), 1), 1.0), 4)


# ── Teacher-feedback weight ───────────────────────────────────────────────────

def _feedback_weight(similarity_score: float) -> float:
    """
    Compute a 0-1 weight based on past teacher confirmations at similar
    similarity levels.  Uses analysis_history.csv loaded lazily.

    If the teacher previously confirmed plagiarism at scores similar to
    *similarity_score* (±15%), the weight nudges upward.
    If previous decisions were mostly Genuine / False Positive at that range,
    the weight nudges downward.
    Returns 0.5 (neutral) when there is no relevant history.
    """
    try:
        from history_manager import load_history
        df = load_history()
        if df.empty:
            return 0.5

        scores = pd.to_numeric(df["similarity_score"], errors="coerce")
        mask   = (scores >= similarity_score - 0.15) & (scores <= similarity_score + 0.15)
        nearby = df[mask]
        if nearby.empty:
            return 0.5

        decisions = nearby["teacher_decision"].str.strip()
        confirmed = (decisions == "Confirmed Plagiarism").sum()
        false_pos = (decisions == "False Positive").sum()
        genuine   = (decisions == "Genuine").sum()
        total     = len(nearby)

        # Lean toward confirmed if majority were confirmed; toward genuine otherwise
        weight = 0.5 + 0.5 * (confirmed - false_pos - genuine) / total
        return round(max(0.0, min(1.0, weight)), 4)
    except Exception:
        return 0.5


# ── AI-assisted writing indicator ────────────────────────────────────────────

def _ai_likelihood_label(
    style_score: float,
    formality: float,
    vocab_richness: float,
) -> str:
    """
    Return "Low", "Medium", or "High" AI-assisted writing likelihood.

    This is a heuristic indicator ONLY — it cannot reliably detect AI.
    It is based on:
      - High formality + high vocabulary richness → possible AI pattern
      - Large style mismatch from prior submissions → possible substitution
    A disclaimer is always shown in the UI alongside this label.
    """
    signals = 0
    if formality      > 0.30:  signals += 1   # very formal prose
    if vocab_richness > 0.80:  signals += 1   # unusually rich vocabulary
    if style_score    > 0.40:  signals += 1   # large style shift from past

    if signals >= 2:
        return "High"
    elif signals == 1:
        return "Medium"
    else:
        return "Low"


# ── Main hybrid analysis entry point ─────────────────────────────────────────

def run_hybrid_analysis(
    new_text: str,
    top_match: dict,
    student_id: str = "",
) -> dict:
    """
    Run the full hybrid plagiarism analysis and return a result dict.

    Parameters
    ----------
    new_text   : the new submission text
    top_match  : the best-match result dict from compare_with_stored()
    student_id : used to look up past submissions for style comparison

    Returns a dict with keys:
        final_score         float 0-1  weighted combined score
        final_risk          str        "🔴 High" / "🟡 Medium" / "🟢 Low"
        phrase_score        float 0-1
        phrase_list         list[str]  matched phrases
        sentence_score      float 0-1
        semantic_score      float 0-1
        semantic_available  bool       Always True (lightweight paraphrase similarity, no optional dep)
        style_score         float 0-1
        style_features_new  dict       style fingerprint of new submission
        feedback_weight     float 0-1
        ai_likelihood       str        "Low" / "Medium" / "High"
        tfidf_score         float 0-1  original TF-IDF baseline score
    """
    matched_text = top_match.get("matched_text", "")
    tfidf_score  = top_match.get("similarity_score", 0.0)

    # ── Component scores ──────────────────────────────────────────────────────
    phrase_score, phrase_list = _phrase_match_score(new_text, matched_text)
    sentence_score  = _sentence_similarity_score(new_text, matched_text, threshold=0.75)
    semantic_score  = _semantic_similarity_score(new_text, matched_text)
    # Lightweight paraphrase similarity — always available (pure scikit-learn, no optional deps)
    sem_for_weight  = semantic_score
    style_score     = _style_mismatch_score(new_text, student_id)
    fb_weight       = _feedback_weight(tfidf_score)

    # ── Weighted final score ───────────────────────────────────────────────────
    final = (
        phrase_score   * _PHRASE_WEIGHT   +
        sentence_score * _SENTENCE_WEIGHT +
        sem_for_weight * _SEMANTIC_WEIGHT +
        style_score    * _STYLE_WEIGHT    +
        fb_weight      * _FEEDBACK_WEIGHT
    )
    final = round(min(final, 1.0), 4)

    # ── Risk label ────────────────────────────────────────────────────────────
    if final >= 0.65:
        final_risk = "🔴 High"
    elif final >= 0.35:
        final_risk = "🟡 Medium"
    else:
        final_risk = "🟢 Low"

    # ── AI-likelihood ─────────────────────────────────────────────────────────
    new_style  = _style_features(new_text)
    ai_label   = _ai_likelihood_label(
        style_score        = style_score,
        formality          = new_style["formality"],
        vocab_richness     = new_style["vocab_richness"],
    )

    return {
        "final_score":        final,
        "final_risk":         final_risk,
        "phrase_score":       phrase_score,
        "phrase_list":        phrase_list,
        "sentence_score":     sentence_score,
        "semantic_score":     semantic_score,
        "semantic_available": True,   # always True — no torch/transformers dependency
        "style_score":        style_score,
        "style_features_new": new_style,
        "feedback_weight":    fb_weight,
        "ai_likelihood":      ai_label,
        "tfidf_score":        tfidf_score,
    }


# ── Submission persistence (extended schema) ──────────────────────────────────

def add_new_submission(
    student_id: str,
    student_name: str,
    assignment_title: str,
    submission_text: str,
    similarity_score: float = 0.0,
    risk_level: str = "",
    teacher_feedback: str = "",
    teacher_notes: str = "",
    granite_summary: str = "",
) -> None:
    """
    Append a student submission to old_submissions.csv with the full extended schema.
    All new fields are optional and default to empty/zero so old callers still work.
    """
    df = load_submissions()
    new_row = pd.DataFrame(
        [
            {
                "student_id":       student_id,
                "student_name":     student_name,
                "assignment_title": assignment_title,
                "submission_text":  submission_text,
                "submission_date":  datetime.now().strftime("%Y-%m-%d"),
                "similarity_score": round(float(similarity_score), 4),
                "risk_level":       risk_level,
                "teacher_feedback": teacher_feedback,
                "teacher_notes":    teacher_notes,
                "granite_summary":  granite_summary,
            }
        ]
    )
    df = pd.concat([df, new_row], ignore_index=True)
    save_submissions(df)
