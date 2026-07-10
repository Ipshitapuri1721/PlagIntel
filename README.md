# PlagIntel 🔍
### AI-Driven Plagiarism Intelligence for Assignments
> Powered by **IBM Granite** on **watsonx.ai** · Built with **Streamlit**

---

## What is PlagIntel?

PlagIntel is an agentic AI web application that helps teachers detect, analyse, and document assignment plagiarism. A teacher pastes or uploads a student submission; the system runs a **5-component hybrid detection pipeline**, scores plagiarism risk, generates a plain-language explanation using **IBM Granite** through watsonx.ai, and records the teacher's decision for continuous improvement.

---

## Features

| Feature | Description |
|---|---|
| 📄 Paste or upload assignment | Accept text directly or upload a **TXT, PDF, or DOCX** file |
| 🔬 Hybrid detection engine | 5-component weighted pipeline: exact phrases, sentence similarity, semantic similarity, writing-style mismatch, teacher feedback weight |
| 📊 Risk score | Colour-coded **High / Medium / Low** risk with full sub-score breakdown |
| 🤖 IBM Granite explanation | 7-section plain-language AI analysis via watsonx.ai |
| 🎨 Three-tier match highlighting | 🟠 exact phrase · 🟡 similar sentence · 🔵 uncommon shared term |
| 🧑‍🏫 Teacher feedback loop | Genuine / Suspicious / Confirmed Plagiarism / False Positive |
| 📈 Dashboard tab | 11 metric cards + 5 Plotly charts with date-range filtering |
| 📉 Analytics tab | Teacher performance, trend analysis, top cases, topic clusters |
| 📥 Export reports | TXT, CSV, and PDF reports with one click |
| 💾 Save & learn | Every decision is stored; past decisions nudge future risk scores |

---

## Agentic Workflow (6 Steps)

```
Step 1 → Accept assignment text (paste or upload TXT / PDF / DOCX)
Step 2 → Compare with stored submissions (hybrid 5-component engine)
Step 3 → Display risk score + 9-point result panel + match highlights
Step 4 → Ask IBM Granite to explain suspicious content (7-section report)
Step 5 → Collect teacher feedback (4 decisions + free-text notes)
Step 6 → Save submission + feedback to database for future improvement
```

---

## Hybrid Detection Engine

The core of PlagIntel is a **5-component weighted pipeline** in [`plagiarism_engine.py`](plagiarism_engine.py):

| Component | Weight | Method |
|---|---|---|
| Exact phrase match | 30% | Sliding-window phrase matching (min 5 words) |
| Sentence-level similarity | 25% | TF-IDF cosine similarity per sentence pair (threshold 0.75) |
| Semantic similarity | 25% | `sentence-transformers` all-MiniLM-L6-v2 (optional — falls back to TF-IDF) |
| Writing-style mismatch | 10% | 4 style features vs student's past submissions |
| Teacher feedback weight | 10% | Past teacher decisions at similar risk scores nudge the final score |

**Final score** = weighted sum of all five components.  
**TF-IDF baseline** is also shown separately for transparency.

> ⚠️ **AI-Assisted Writing Likelihood** is a heuristic estimate only. It is not a definitive determination of AI authorship.

---

## IBM Granite Explanation (7 Sections)

When the teacher clicks **"Ask IBM Granite"**, the system calls `ibm/granite-4-h-small` on watsonx.ai with a structured prompt containing the full hybrid scores. Granite returns a 7-section report:

1. **Case Summary** — overall assessment at a glance
2. **Reason for Similarity** — why the scores are elevated
3. **Matching Concepts** — shared themes or arguments
4. **Suspicious Matching Phrases** — only phrases present in both texts (no hallucination)
5. **Potential Cause** — paraphrase, shared source, coincidence, AI-assisted writing
6. **Copy Type** — Exact Copy / Paraphrased / Mosaic / AI-Assisted Writing / Low Concern
7. **Teacher Recommendation** — suggested action + confidence score (0–100)

Granite is instructed to **only reference phrases actually present in the submitted texts** and never fabricate content.

---

## Three-Tier Match Highlighting

Matched text in the top-matching submission is highlighted with three tiers:

| Tier | Colour | Meaning |
|---|---|---|
| 🟠 Orange solid | `#ff6600` | Exact phrase (≥ 3 words) copied verbatim |
| 🟡 Yellow dashed | `#ffd700` | Semantically similar sentence |
| 🔵 Blue dotted | `#0f62fe` | Uncommon shared term (not a stopword) |

---

## Project Structure

```
PlagIntel/
│
├── app.py                  # Streamlit UI — 3 tabs, full agentic workflow, IBM Blue theme
├── plagiarism_engine.py    # Hybrid 5-component detection engine + TF-IDF baseline + highlighting
├── granite_explainer.py    # IBM Granite prompt builder + watsonx.ai API call + offline fallback
├── history_manager.py      # analysis_history.csv CRUD + feedback stats + learning context
├── dashboard.py            # Dashboard tab — 11 metric cards, 5 Plotly charts, search/filter
├── analytics.py            # Analytics tab — teacher performance, trends, top cases
├── report_generator.py     # TXT / CSV / PDF report generation (reportlab + stdlib fallback)
├── requirements.txt        # Python dependencies
│
└── data/
    ├── old_submissions.csv   # Permanent submission database (10-column schema)
    └── analysis_history.csv  # Runtime review history (created on first save)
```

---

## Setup & Installation

### 1. Clone / Download

```bash
git clone https://github.com/your-org/plagintel.git
cd plagintel
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB) on first run.  
> If you skip it, the semantic similarity component falls back to TF-IDF automatically.

### 4. Configure IBM watsonx.ai credentials

Create a `.env` file in the project root:

```env
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

You can obtain these from the [IBM watsonx.ai console](https://dataplatform.cloud.ibm.com).

> If credentials are missing or invalid, the app falls back to a **local offline explanation** so the workflow is never blocked.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Data Files

### `data/old_submissions.csv` (10-column schema)

| Column | Description |
|---|---|
| `student_id` | Unique student identifier |
| `student_name` | Student's full name |
| `assignment_title` | Assignment title |
| `submission_text` | Full text of the submission |
| `submission_date` | ISO date string |
| `subject` | Subject / course |
| `grade_level` | Grade or year level |
| `word_count` | Word count of the submission |
| `teacher_id` | Teacher who reviewed it |
| `feedback_label` | Teacher's decision label |

### `data/analysis_history.csv` (11-column schema, runtime)

| Column | Description |
|---|---|
| `timestamp` | ISO datetime of the review |
| `student_id` | Student identifier |
| `student_name` | Student name |
| `assignment_title` | Assignment title |
| `similarity_score` | Final hybrid score (0–1) |
| `risk_level` | High / Medium / Low |
| `top_matching_student` | Name of the closest matching submission |
| `top_matching_assignment` | Title of the closest matching submission |
| `granite_summary` | Full Granite explanation text |
| `feedback` | Teacher decision |
| `notes` | Teacher free-text notes |

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `WATSONX_API_KEY` | Yes | IBM Cloud API key |
| `WATSONX_PROJECT_ID` | Yes | watsonx.ai project ID |
| `WATSONX_URL` | No | Defaults to `https://us-south.ml.cloud.ibm.com` |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.35.0 | Web UI framework |
| `pandas` | 2.2.2 | Data handling (CSV I/O) |
| `scikit-learn` | 1.4.2 | TF-IDF vectoriser + cosine similarity |
| `ibm-watsonx-ai` | 1.0.10 | IBM Granite API client |
| `python-dotenv` | 1.0.1 | `.env` file loader |
| `PyMuPDF` | 1.24.5 | PDF text extraction |
| `python-docx` | 1.1.2 | DOCX text extraction |
| `plotly` | 5.22.0 | Interactive charts (Dashboard + Analytics) |
| `reportlab` | 4.2.2 | PDF report generation |
| `sentence-transformers` | 2.7.0 | Semantic similarity (optional — falls back to TF-IDF) |

---

## Offline / Fallback Behaviour

| Scenario | Fallback |
|---|---|
| No watsonx.ai credentials | Local heuristic explanation generated from scores |
| `sentence-transformers` not installed | Semantic slot uses TF-IDF cosine similarity |
| `reportlab` not installed | PDF generated via Python stdlib binary writer |
| Empty `analysis_history.csv` | Dashboard and Analytics tabs show zero-state gracefully |

---

## Teacher Feedback Options

| Decision | Meaning |
|---|---|
| ✅ Genuine | Original work — no concern |
| ⚠️ Suspicious | Needs further review |
| 🚨 Confirmed Plagiarism | Formally flagged for action |
| ❌ False Positive | System over-flagged; overrides future scoring |

Past decisions are injected into the Granite prompt as **learning context**, so the AI explanation improves over time as more decisions are recorded.

---

## Notes for Educators

- **Risk scores are indicators, not verdicts.** Always review the highlighted text and Granite explanation before making a decision.
- **AI-Assisted Writing Likelihood** is a heuristic based on style-feature deviation. It does not identify specific AI tools and should not be used as proof of AI authorship.
- **False Positive feedback** is especially valuable — it teaches the system to down-weight similar patterns in future analyses.
- All data is stored locally in CSV files. No student data is sent to IBM watsonx.ai; only the assignment **text** is submitted for analysis.

---

## Licence

MIT — free for educational and research use.

---

*PlagIntel · Powered by IBM Granite on watsonx.ai · Built with Streamlit*
