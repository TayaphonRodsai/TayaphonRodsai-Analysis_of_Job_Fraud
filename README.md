# Detecting Fraudulent Job Postings with NLP

Text classification project that flags fake job advertisements, with a focused
evaluation on healthcare postings and a test against live job ads from a public API.

Course project — *MDSSB-MET-02 Text Analysis and Natural Language Processing*,
M.Sc. Data Science for Society and Business, Constructor University Bremen.

---

## Why this problem

Job scams target people who are already in a weak position: job seekers, and often
migrant or care workers. Scam ads look like normal ads. The signal is in the wording,
not in the metadata. That makes it a text problem.

The extra difficulty is imbalance. Only **4.8%** of the postings in the dataset are
fraudulent, so a model that predicts "real" every time is already 95% accurate and
completely useless. Recall on the fraud class is the metric that matters.

---

## Data

**EMSCAD** — the Employment Scam Aegean Dataset, published as
[Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
on Kaggle.

| | Count | Share |
|---|---:|---:|
| Legitimate | 17,014 | 95.2% |
| Fraudulent | 866 | 4.8% |

**The CSV is not in this repository.** Download `fake_job_postings.csv` from the link
above and put it next to the notebook before running.

Live job ads come from the [Adzuna API](https://developer.adzuna.com/) (free tier).
That data is not included either.

---

## Method

1. **Clean** — lowercase, strip HTML tags, digits and punctuation from the job description.
2. **Vectorise** — TF-IDF, 3,000 features, unigrams and bigrams, English stop words removed.
3. **Split** — 80/20, stratified on the fraud label, `random_state=2026`.
4. **Baseline** — Multinomial Naive Bayes.
5. **Main model** — Random Forest with `class_weight="balanced"`, tuned by `GridSearchCV`
   (5-fold, scored on F1). Best parameters: `n_estimators=50`, `max_depth=None`.
6. **Evaluate** — on the full test set, then on a healthcare-only subset.
7. **Apply** — score 200 live UK healthcare ads pulled from Adzuna.

---

## Results

### Full test set (3,576 postings, 173 fraudulent)

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) | Accuracy |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.98 | 0.28 | 0.43 | 0.96 |
| **Random Forest** | **0.92** | **0.61** | **0.74** | **0.98** |

Naive Bayes is almost never wrong when it says "fraud", but it misses 72% of the scams.
The Random Forest more than doubles recall at a small cost in precision. For this
problem that is the right trade — a missed scam hurts a person, a false alarm only
costs a second look.

The Random Forest still misses **67 scams** out of 173.

### Healthcare subset

The subset was built in two steps, not guessed. A seed search on domain words
(`nurse`, `hospital`, `clinic`, `medical`, `healthcare`) returned 1,412 posts. Unigram
and bigram frequency analysis on that subset showed which terms were actually
healthcare-specific — the top unigrams were mostly generic job words (*experience*,
*work*, *team*), while the bigrams were far more useful (*home health*, *patient care*,
*health care*). The final keyword list mixes data-discovered terms with domain terms.

Final healthcare subset: **1,404 posts, 106 fraudulent (7.5%)** — noticeably higher than
the 4.8% base rate.

On the healthcare slice of the test set (303 posts, 23 fraudulent):

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) | Accuracy |
|---|---:|---:|---:|---:|
| Naive Bayes | 1.00 | 0.35 | 0.52 | 0.95 |
| Random Forest | 0.93 | 0.57 | 0.70 | 0.96 |

Recall drops from 0.61 to 0.57. The model is slightly worse exactly where the fraud
rate is highest. That is the honest finding of this project.

### Live data

200 UK healthcare ads were pulled from Adzuna and scored with the saved model. Almost
all scored low. Since these ads have no ground-truth labels, the output is a
demonstration that the pipeline runs end to end — not a verdict on any employer.

---

## A note on ethics

**Employer names in this notebook are anonymised** (`Company A`, `Company B`, …).

The model has 61% recall and 92% precision. It is wrong often enough that publishing a
real company's name next to a fraud score would be unfair and potentially damaging.
Nothing in this repository should be read as an accusation against any real business.

The EMSCAD labels come from the dataset authors, not from me.

---

## Running it

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib requests
```

1. Download `fake_job_postings.csv` (see **Data**) into this folder.
2. For the live-data section, get a free Adzuna `app_id` and `app_key` and fill them
   into the API cell. Do not commit them.
3. Open `Analysis_of_Job_Fraud_detection_public.ipynb` and run the cells in order.

Training writes `fraud_model.pkl` and `tfidf_vectorizer.pkl`. Both are ignored by git —
re-run the training cell to regenerate them.

---

## What I would do next

- **Use more than the description.** Only `description` is vectorised. The dataset also
  has `company_profile`, `requirements` and `benefits`, and missing fields are
  themselves a signal — a real employer usually fills in a company profile.
- **Handle the imbalance directly.** `class_weight="balanced"` is a start; SMOTE or
  threshold tuning on the predicted probability would likely lift recall further.
- **Try a transformer.** TF-IDF cannot see word order or context. A fine-tuned DistilBERT
  is the obvious comparison.
- **Train on the healthcare subset itself,** rather than applying a general model to it.

---

## Files

| File | What it is |
|---|---|
| `Analysis_of_Job_Fraud_detection_public.ipynb` | Full analysis — EDA, pipeline, healthcare evaluation, live API test |
| `.gitignore` | Keeps data files and trained models out of the repo |
| `LICENSE` | MIT (code only) |

---

**Tayaphon Rodsai** — [github.com/TayaphonRodsai](https://github.com/TayaphonRodsai)

Code is MIT licensed. The EMSCAD dataset and the Adzuna data keep their own terms.
