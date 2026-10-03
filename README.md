# ExamLens: AI PDF-Based Subjective Exam Evaluator

An academic NLP project that turns a study PDF into an interactive subjective exam workflow.

This first version is being built with a modular architecture so the core NLP pipeline remains visible and explainable:

- PDF upload and extraction
- preprocessing
- topic and keyword analysis
- question generation
- answer evaluation
- summarization
- FastAPI backend
- Web UI (vanilla HTML/CSS/JS served by FastAPI, no build step)

## Project Goal

The application helps a student:

1. Upload a study PDF
2. Extract and preprocess text
3. Identify important concepts
4. Generate subjective questions
5. Answer questions one by one
6. Receive an explainable evaluation score and feedback

The project is designed so the scoring is not a black box. NLP methods such as TF-IDF, bag of words, n-grams, semantic similarity, and embeddings remain visible in the implementation.

## Planned NLP Pipeline

`PDF -> extraction -> preprocessing -> BoW/TF-IDF -> topic extraction -> question generation -> answer evaluation -> feedback`

Transformer-based models are planned only for supportive tasks such as question generation and summarization, not as the sole scoring mechanism.

## Initial Architecture

```text
app/
  api/                FastAPI application and request/response models
  web/                Single-page web UI (index.html + static/app.css, app.js)
  services/           Orchestration between NLP, PDF, and evaluation layers
  nlp/                Preprocessing, representation, analysis, embeddings
  pdf/                PDF extraction and validation
  evaluation/         Explainable answer scoring logic
  question_generation/ Topic-to-question generation
  utils/              Shared helpers, config, logging
tests/                Unit tests for important NLP components
data/                 Sample documents and intermediate assets
models/               Local model cache or lightweight artifacts
```

## NLP Concepts to Demonstrate

- Tokenization
- Stop-word removal
- Lemmatization
- Bag of Words
- TF-IDF
- N-grams
- POS tagging
- Named Entity Recognition
- Sentence/document embeddings
- Semantic similarity
- Transformers
- Summarization
- Text generation
- REST API deployment

## NLP Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Tokenization | Regex tokenization in `app/nlp/preprocessing.py` |
| Stop-word removal | Explicit educational stop-word set |
| Lemmatization | Lightweight suffix-based lemma baseline |
| BoW | Lemma frequency dictionary |
| TF-IDF | scikit-learn `TfidfVectorizer`, with a fallback |
| Unigrams, bigrams, trigrams | Frequency-ranked n-gram extraction |
| POS tagging | NLTK averaged perceptron (`app/nlp/linguistics.py`), colour-coded in the UI |
| Lemmatization | POS-aware WordNet lemmatizer (suffix baseline as fallback) |
| NER | Heuristic by default; opt-in local BERT NER (`dslim/bert-base-NER`) |
| TextRank | From-scratch PageRank for summarization and keyphrases (`app/nlp/textrank.py`) |
| Retrieval / QA | BM25 (from scratch) + dense embeddings fused by Reciprocal Rank Fusion (`app/nlp/retrieval.py`) |
| Explainable scoring | Reference ideas aligned to the student's sentences (`align_ideas`) |
| Embeddings | Optional local `all-MiniLM-L6-v2`; lexical fallback when unavailable |
| Semantic similarity | Embedding cosine similarity or token-set baseline |
| Text generation | Deterministic topic-to-question templates |
| Summarization | Extractive TextRank summary |
| REST API | FastAPI endpoints under `/api` |

## What's New

- **Ask the Document** tab: natural-language search with the answer sentence highlighted and a BM25/semantic score breakdown.
- **NLP Analysis** tab is now a visual lab: coloured POS tags, tokens to lemmas, TF-IDF vs TextRank side by side, a concept map graph, sentence-importance ranking and optional BERT NER.
- **Exam feedback** shows an idea-by-idea check: which reference ideas your answer expressed, and which sentence expressed them.
- The embedding model is loaded once and cached instead of on every call.

- **Contradiction detection**: a local NLI model (`app/nlp/nli.py`) catches answers that reuse the right words but say the wrong thing (e.g. "TF-IDF *ignores* term frequency"); these are penalised and explained.
- **Retrieve + rerank**: search runs at sentence level with BM25 + embeddings (RRF), then a cross-encoder re-ranks the best candidates.
- **Results tab**: score chart, per-component table, a "Revise these" plan that retrieves the PDF passage for every missed concept, and a downloadable Markdown study report (`app/services/report.py`).
- **Bug fix**: PDF upload previously crashed (`ExtractedDocument` was never defined); it is fixed and covered by `tests/test_pdf_upload.py`.

## Scoring Methodology

The result is an `Evaluation Score`, not objective truth. The default formula is:

```text
overall = 0.50 * semantic_similarity
        + 0.30 * concept_coverage
        + 0.20 * keyword_coverage
overall *= 1 - 0.6 * (contradicted ideas / reference ideas)   # NLI penalty
```

Each component is normalized to 0–100. Semantic similarity uses a local sentence-transformer when installed and falls back to lexical Jaccard similarity. Concept coverage checks whether the answer mentions each reference concept, while keyword coverage checks normalized reference terms. The weights are centralized in `app/evaluation/scoring.py` so they can be adjusted for experiments.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Open http://127.0.0.1:8000 for the web app (`./run.sh` also works). The REST API documentation is at `http://127.0.0.1:8000/docs`.

The application also supports direct text analysis through `/api/analyze`, which is useful for testing without a PDF. Uploaded PDFs are validated for type, size, emptiness, and extractable text. Scanned PDFs currently receive a clear OCR-not-enabled error.

## API Endpoints

- `GET /api/health`
- `POST /api/upload` with multipart PDF file
- `POST /api/analyze` with `{ "text": "...", "filename": "..." }`
- `POST /api/questions` with text, topics, count, and difficulty
- `POST /api/evaluate` with answer, reference answer, concepts, and keywords
- `POST /api/summarize` with text
- `POST /api/search` with a query and chunks (hybrid BM25 + semantic)
- `POST /api/keyphrases` with text (TextRank)

## Project Structure

```text
app/
  api/main.py                 FastAPI routes and Pydantic request models
  web/                        Web UI: document, NLP insights, search, exam, results
  services/lab.py             data for the NLP Insights view (graph, TextRank, POS...)
  question_generation/concept_generator.py   mines defined concepts, asks Easy/Medium/Hard questions
  pdf/extraction.py           PDF validation and PyMuPDF extraction
  nlp/preprocessing.py        tokens, lemmas, BoW, TF-IDF, n-grams, POS/NER baseline
  nlp/embeddings.py           optional local sentence embeddings
  services/pipeline.py        topics, summary, and serialization orchestration
  question_generation/        template-based subjective questions
  evaluation/scoring.py       explainable weighted scoring and feedback
tests/                        unit tests for the core pipeline
```

## Limitations and Future Work

The first version prioritizes transparency and laptop-friendly behavior. POS and NER are intentionally lightweight baselines; a future phase can load spaCy's English model when available. OCR for scanned PDFs, persisted exam sessions, richer section detection, and a locally cached transformer question generator are natural extensions. No external LLM API is used for core scoring.

## Notes

- No paid API is required for the core scoring logic.
- The project will be kept lightweight enough to run on a normal student laptop.
- The score will be presented as an `Evaluation Score`, not as absolute truth.
