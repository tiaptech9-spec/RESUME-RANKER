# ResumeRank — AI Resume Screening System

## Dataset

The dataset used for this project is available on Kaggle:

- **Dataset:** Resume Dataset
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/palaksood97/resume-dataset

Download the dataset from Kaggle and place the CSV file in the project directory before running the application.

A Flask-based AI Resume Ranker implemented around the ML workflow in `backend.ipynb`.

## Core ML pipeline
1. Read DOCX/PDF resumes.
2. Extract paragraph and table text.
3. Clean and normalize text.
4. TF-IDF using English stopword removal, 1–2 grams and max 10,000 features.
5. Semantic embeddings using `all-MiniLM-L6-v2`.
6. Cosine similarity for both representations.
7. Final score = 50% TF-IDF + 50% semantic similarity.
8. Rank candidates and assign: Shortlisted (>=70), Under Review (50–69.99), Rejected (<50).

## Functional features
- Single/multiple DOCX or PDF resume upload.
- Job description paste or DOCX/PDF/TXT upload.
- Resume parsing: name, email, phone, skills, qualification and experience.
- Required/matched/missing skill analysis.
- Candidate ranking and score breakdown.
- Search by name/email/filename/skill/resume text.
- Filters for score, status, experience, skill and qualification.
- View original resume and download it.
- Full candidate CRUD: create through screening, read, update recruiter fields, delete.
- Recruiter status and feedback loop: Pending/Hired/Rejected/Maybe.
- Feedback-based retraining model after sufficient labeled feedback.
- Excel and PDF export.
- Bulk processing and responsive interface.
- Parameterized SQLite queries and upload type/size validation.

## Run
```bash
python -m venv venv
# Windows: venv\\Scripts\\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000`.

The first semantic-screening run can take longer because the sentence-transformers model may need to download/load. Later runs are faster.

## Project files
- `app.py` — Flask routes, parsing and ML pipeline
- `database.py` — SQLite schema and CRUD/data operations
- `templates/` — UI pages
- `static/style.css` — separate styling
- `uploads/` — uploaded resumes will be shown when file is executed
- `resume_ranker.db` — local database will be created after file execution
- `backend.ipynb` — original notebook workflow
- `requirements.txt` — dependencies
- `AI Resume Ranker Documentation.docx` — documentation
- `AI RESUME RANKER blog.docx` — blog source

## Notes
The web application's semantic scoring follows the notebook model. A safe keyword-overlap fallback is used only if the sentence-transformers package/model is unavailable, so the local UI remains usable.
