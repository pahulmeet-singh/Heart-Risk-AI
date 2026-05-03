# Heart Disease Clinical Risk Assessment
## INT428 – AI Essentials | College Project

---

## Quick Start

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn joblib matplotlib

# Run the app (from this folder)
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## Project Structure

```
heart_project/
├── app.py                      ← Streamlit web app (FIXED — see bugs below)
├── EDA.ipynb                   ← Jupyter notebook: EDA + model training (FIXED)
├── heart.csv                   ← Cleveland Heart Disease dataset (303 rows)
├── heart_disease_model.pkl     ← Saved Logistic Regression model
├── heart_scaler.pkl            ← Saved StandardScaler
├── model_columns.pkl           ← Saved feature name order
├── Attribute_Description.xlsx  ← Feature documentation
└── README.md                   ← This file
```

---

## Bugs Fixed (Critical — Know These for Viva)

### Bug 1: One-Hot Encoding Mismatch

**What was wrong:**
```python
# ❌ OLD (broken)
df_encoded = pd.get_dummies(df_input, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
```
The model was trained on raw integer labels. Adding `pd.get_dummies` created columns
like `cp_1`, `cp_2` that the model had never seen. The `reindex` silently filled them
with **0**, making every categorical input completely ignored.

**Fix:**
```python
# ✅ FIXED
df['oldpeak_log'] = np.log1p(df['oldpeak'])
df.drop('oldpeak', axis=1, inplace=True)
df = df[model_columns]
scaled = scaler.transform(df)
```

---

### Bug 2: Inverted Target Class Index

**What was wrong:**
```python
# ❌ OLD (inverted)
risk_prob = model.predict_proba(scaled_data)[:, 1][0] * 100
```
In this Kaggle dataset, `target=0` = disease and `target=1` = healthy.
So `predict_proba[:, 1]` returns the probability of being **healthy**, not sick.
This meant every high-risk patient showed a LOW percentage and vice versa.

**Fix:**
```python
# ✅ FIXED — P(class=0) = P(disease)
risk_prob = model.predict_proba(scaled_data)[:, 0][0] * 100
```

---

## How the Model Works (Viva Cheat Sheet)

### Algorithm
**Logistic Regression** — computes a weighted sum of features, passes it through
a sigmoid function to produce a probability between 0 and 1.

```
P(disease) = sigmoid(b0 + b1*age + b2*sex + ... + b13*oldpeak_log)
```

### Full Preprocessing Pipeline

| Order | Step | Why |
|-------|------|-----|
| 1 | `drop_duplicates()` | 1 duplicate found in dataset |
| 2 | Cap `chol` at 99th percentile | Extreme cholesterol outliers skew the model |
| 3 | `log1p(oldpeak)` → `oldpeak_log` | oldpeak is right-skewed; log reduces this |
| 4 | Drop original `oldpeak` | Avoid multicollinearity with `oldpeak_log` |
| 5 | Z-score cap (±3σ) on numeric cols | Clips remaining outliers without removing data |
| 6 | `StandardScaler` | Logistic Regression is sensitive to feature scale |
| 7 | Train/test split 80/20 | Evaluate on unseen data |

### Why No One-Hot Encoding?
Categorical features (`cp`, `slope`, `ca`, `thal`, `restecg`) are kept as integers.
The model treats them as ordinal. This simplified the pipeline and the `.pkl` files
reflect this choice — the app must match it exactly.

---

## Feature Reference

| Feature | Values | Clinical Meaning |
|---------|--------|-----------------|
| `age` | 29–77 | Age in years |
| `sex` | 0=Female, 1=Male | Biological sex |
| `cp` | 0=Typical Angina → 3=Asymptomatic | 0 most risky in this dataset |
| `trestbps` | mm Hg | Resting blood pressure |
| `chol` | mg/dl | Serum cholesterol |
| `fbs` | 0/1 | Fasting blood sugar > 120 mg/dl |
| `restecg` | 0–2 | Resting ECG result |
| `thalach` | bpm | Maximum heart rate achieved |
| `exang` | 0/1 | Chest pain during exercise |
| `oldpeak` | 0–6.2 | ST depression (ECG measure) |
| `slope` | 0–2 | Slope of ST segment |
| `ca` | 0–3 | Number of major vessels visible |
| `thal` | 1–3 | 3=Reversible defect (most concerning) |
| `target` | 0=Disease, 1=Healthy | **0 is disease in this dataset** |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~85% |
| AUC-ROC | ~0.90 |
| Train size | 241 rows |
| Test size | 61 rows |

---

*This tool is for educational purposes only. Not for clinical use.*
