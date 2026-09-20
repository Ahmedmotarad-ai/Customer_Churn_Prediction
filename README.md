# Customer Churn Analysis and Prediction — Telco

Saiket Systems ML internship project: analyze customer churn for a
telecommunications company and serve a predictive model behind an API.
Work covers all 6 project tasks (only 4 required) plus production
packaging (API, Docker, Compose, tests).

## 1. Business problem

telecommunications company loses ~26% of its customers. The goal: predict which
customers are about to leave so the company can retain them with targeted
offers. Success means high **recall on churners** (catch them before they
leave), not just high overall accuracy.

## 2. Dataset

`tasks/Telco_Customer_Churn_Dataset.csv` — 7043 customers × 21 columns.

### 2.1 Data dictionary

| Column | Type | Values / range |
|---|---|---|
| customerID | id | unique per row (dropped before modeling) |
| gender | cat | Male (3555), Female (3488) |
| SeniorCitizen | 0/1 | 0: 5901, 1: 1142 |
| Partner | cat | No 3641 / Yes 3402 |
| Dependents | cat | No 4933 / Yes 2110 |
| tenure | int | 0–72 months, mean 32.4 |
| PhoneService | cat | Yes 6361 / No 682 |
| MultipleLines | cat | No 3390 / Yes 2971 / No phone service 682 |
| InternetService | cat | Fiber optic 3096 / DSL 2421 / No 1526 |
| OnlineSecurity | cat | No 3498 / Yes 2019 / No internet service 1526 |
| OnlineBackup | cat | No 3088 / Yes 2429 / No internet service 1526 |
| DeviceProtection | cat | No 3095 / Yes 2422 / No internet service 1526 |
| TechSupport | cat | No 3473 / Yes 2044 / No internet service 1526 |
| StreamingTV | cat | No 2810 / Yes 2707 / No internet service 1526 |
| StreamingMovies | cat | No 2785 / Yes 2732 / No internet service 1526 |
| Contract | cat | Month-to-month 3875 / Two year 1695 / One year 1473 |
| PaperlessBilling | cat | Yes 4171 / No 2872 |
| PaymentMethod | cat | Electronic check 2365 / Mailed check 1612 / Bank transfer 1544 / Credit card 1522 |
| MonthlyCharges | float | 18.25–118.75, mean 64.76 |
| TotalCharges | float | 0–8684.8, mean 2279.7 (after fix) |
| Churn (target) | cat | No 5174 (73.5%) / Yes 1869 (26.5%) |

### 2.2 Notebook walkthrough (`tasks/`, run in order 1 → 6)

Every file is standalone (setup cell on top; CSV exports at the bottom
where state changes). `tasks/data/` carries state between levels.

**Task1_Data_Preparation.ipynb** (16 code cells + setup/export)
1. Imports (numpy, pandas, matplotlib, seaborn, sklearn pieces).
2. Loads the CSV; `head/info/shape` → (7043, 21).
3. `isnull().sum()` → 0 everywhere (blanks are hidden as `" "`).
4. Blank-string scan → 11 blanks, all in TotalCharges.
5. Displays the 11 rows; counts `tenure == 0` → also 11 (same customers).
6. Markdown finding: new customers, TotalCharges must be 0.
7. Fix: `pd.to_numeric(errors="coerce")` + fill 0 where tenure == 0
   (NaNs 11 → 0).
8. Duplicates (0 rows, 0 IDs), numeric describe + IQR outlier scan
   (0 outliers everywhere), categorical value_counts, target/SeniorCitizen
   distribution, customerID drop → (7043, 20).
9. Boxplots + histograms of the 3 numerics.
10. Export `data/df_task1.csv`.

**Task2_Train_Test_Split.ipynb** (setup + 2 task cells + export)
1. Setup loads `df_task1.csv`.
2. `X = df.drop(["Churn", "gender"])`, `y = (Churn == "Yes")`; stratified
   80/20 split (`random_state=42`) → train (5634, 18), test (1409, 18),
   churn rate 26.54% preserved in both.
3. `ColumnTransformer`: StandardScaler on
   [tenure, MonthlyCharges, TotalCharges, SeniorCitizen] + OneHotEncoder
   (`handle_unknown="ignore"`) on the other 14 → 43 features, fit on
   train only.
4. Export `X, y, X_train, X_test, y_train, y_test` to `data/`.

**Task3_Feature_Selection.ipynb** (setup + 3 cells + export)
1. Numeric means by churn: tenure 37.6 vs 18.0 months; MonthlyCharges
   61.3 vs 74.4; TotalCharges 2550 vs 1532. Correlations with churn:
   tenure −0.35, MonthlyCharges +0.19, TotalCharges −0.20.
2. Churn % per category — biggest gaps: Contract Month-to-month 42.7%
   vs Two year 2.8%; Fiber optic 41.9% vs No-internet 7.4%; Electronic
   check 45.3%; no-TechSupport 41.6% vs with 15.2%; gender flat
   (26.9% vs 26.2%).
3. Bar plots (Contract, InternetService) + tenure histogram by churn.
4. Decision: drop `gender`; export `data/df_task3.csv` (with ChurnFlag).

**Task4_Model_Selection.ipynb** (setup + 1 cell)
- (A) LogisticRegression baseline: fast, interpretable, loves scaled
  one-hot data; `class_weight="balanced"`, `max_iter=1000`.
- (B) RandomForest comparator: captures non-linear interactions
  (Contract × tenure); `n_estimators=200`, `class_weight="balanced"`.
- Both wrapped as `Pipeline(preprocessor + model)`; winner decided in
  Task 6 on the five required metrics.

**Task5_Model_Training.ipynb** (setup + 1 cell)
- Fits both pipelines on `X_train`. Train accuracy: LR 0.75, RF ~1.00
  (RF memorizes train — overfit flag, judged on test next).

**Task6_Model_Evaluation.ipynb** (setup + 5 cells + conclusion)
1. Five metrics + confusion matrices on test (table in §4).
2. ROC curves (LR above RF).
3. Feature importance: LR coefficients × encoded names (table in §4).
4. 5-fold StratifiedKFold CV on full pipelines (accuracy + ROC-AUC).
5. GridSearchCV (3-fold ROC-AUC): `C ∈ {0.1, 1, 10}` for LR,
   `max_depth ∈ {None, 10, 20}` for RF.
6. Conclusion: LogReg wins (recall 0.78 vs 0.48, AUC 0.842 vs 0.821).

## 3. Decisions log (why, with numbers)

| Decision | Reason |
|---|---|
| TotalCharges blanks → 0 | All 11 blanks are `tenure == 0` customers; median (1397) reflects 2-year customers and would fabricate charges |
| Drop customerID | unique identifier, zero signal |
| Drop gender | churn gap 0.76pp (26.92 vs 26.16) — noise |
| Keep `No internet service` levels | real information (no service), 1526 customers each |
| OneHot, not LabelEncoding | no ordering between values (would inject false ranking) |
| StandardScaler for 4 numerics | equalize tenure/charges scales for LR |
| Stratified 80/20, seed 42 | preserves 26.54% churn in both sets, reproducible |
| Fit preprocessor on train only | avoids test leakage (also inside CV folds) |
| Final = LogReg, not RF | recall 0.783 vs 0.484; business goal is catching churners |
| Production trains on full data | CV already validated; more data for the served model |

## 4. Results

### 4.1 Test set (1409 rows)

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.737 | 0.503 | **0.783** | **0.613** | **0.842** |
| RandomForest | 0.779 | 0.622 | 0.484 | 0.537 | 0.821 |

Confusion matrices — LogReg `[[746, 289], [81, 293]]`,
RF `[[916, 119], [193, 181]]` (rows = actual No/Yes).

### 4.2 Stability + tuning

- 5-fold CV: LogReg acc 0.746±0.006, AUC 0.845±0.013; RF acc
  0.789±0.010, AUC 0.821±0.013 → gap is real, not split luck.
- Tuning: LogReg best `C=10` (AUC 0.846, marginal); RF best
  `max_depth=10` (AUC 0.843, fixes most overfit). Verdict unchanged.

### 4.3 Top drivers (LogReg |coef|)

tenure −1.15 · Contract_Two year −0.78 · Fiber optic +0.70 ·
MonthlyCharges −0.67 · Month-to-month +0.65 · DSL −0.63 ·
TotalCharges +0.49. Highest-risk profile: month-to-month + fiber +
electronic check + short tenure + no TechSupport/OnlineSecurity.

## 5. API (`api/`)

- `train.py` — reproduces the cleaning, trains LogReg (C=10) on the full
  dataset (7043×18), saves `model.pkl` (pipeline + feature order).
- `app.py` — FastAPI app:
  - `GET /health` → `{"status": "ok", "model_loaded": true}`
  - `POST /predict` with 18 customer fields (Pydantic-validated:
    tenure ≥ 0, charges ≥ 0, SeniorCitizen ∈ {0,1}) →
    `{"churn": 1, "churn_label": "Yes", "churn_probability": 0.8193}`
  - Missing/negative fields → 422; unseen categories tolerated
    (`handle_unknown="ignore"`); missing `model.pkl` → clear 500 hint.

```bash
cd api
python train.py
uvicorn app:app --reload        # docs: http://127.0.0.1:8000/docs
```

## 6. Docker

`api/Dockerfile` (build context = repo root): python:3.11-slim →
install `api/requirements.txt` (pinned: fastapi 0.141.1, uvicorn 0.53.0,
scikit-learn 1.8.0, pandas 3.0.3, pydantic 2.13.5, joblib 1.5.3) →
copy code + dataset → `RUN python train.py` (fresh model baked in) →
serve on 8000. `.dockerignore` keeps notebooks/checkpoints/stale CSVs
out of the image. `docker-compose.yml` runs it as service `api`
(port 8000, restart unless-stopped).

```bash
docker build -f api/Dockerfile -t churn-api .
docker run -d --name churn-api -p 8000:8000 churn-api
docker compose up -d --build    # compose-managed alternative
docker compose down             # stop
```

`/health` and `/predict` verified working inside the container
(same 0.8193 as local — container model identical).

## 7. Tests

`api/tests/` — 10 pytest tests, all passing
(`python -m pytest api/tests -v`, ~1.5 s):

| File | Test | Asserts |
|---|---|---|
| test_api | health | 200, status ok, model loaded |
| test_api | predict_shape | keys, label↔class consistency, proba ∈ [0,1] |
| test_api | known_churner | real row → churn 1, proba > 0.5 |
| test_api | low_risk_stays | safe profile → churn 0, proba < 0.5 |
| test_api | negative_tenure_rejected | 422 |
| test_api | missing_field_rejected | 422 |
| test_api | unknown_category_tolerated | 200, bounded proba |
| test_model | artifact_structure | steps [preprocessor, model], 18 features, no gender |
| test_model | probabilities_bounded | 50 real rows, proba ∈ [0,1], rows sum to 1 |
| test_model | high_above_low | high-risk proba (0.62) > low-risk (0.01) |

## 8. Reproduce everything

```bash
# 1. notebooks (Jupyter server rooted here), in order 1 → 6
jupyter notebook tasks/Task1_Data_Preparation.ipynb   # ... through Task6
# 2. API
cd api && python train.py && uvicorn app:app --reload
# 3. tests
python -m pytest api/tests -v
# 4. docker
docker compose up -d --build
```

Tech stack: Python 3.11, scikit-learn 1.8.0, pandas 3.0.3, FastAPI
0.141.1, uvicorn 0.53.0, pydantic 2.13.5, Docker 29.8.

## 9. Layout

```
├── Customer_Churn_Prediction.ipynb   # combined notebook (backup)
├── SKS Machine Learning Task.pdf
├── README.md
├── docker-compose.yml
├── .dockerignore
├── api/
│   ├── app.py  train.py  requirements.txt  Dockerfile  model.pkl
│   └── tests/ (conftest, test_api, test_model)
└── tasks/
    ├── Telco_Customer_Churn_Dataset.csv
    ├── Task1_..._Task6_*.ipynb
    └── data/ (df_task1, df_task3, X, y + train/test splits)
```

## 10. Submission notes

- All 6 tasks complete (requirement: any 4).
- Remaining per the task PDF: LinkedIn showcase video (tag SaiKet
  Systems, hashtags #saiketsystems #saiket #saiketsys) + submission form
  (to be shared).
