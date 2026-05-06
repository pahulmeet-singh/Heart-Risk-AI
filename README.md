# 🫀 Heart Disease Clinical Risk Assessment
### INT428 – AI Essentials | Deployed ML Project

A logistic regression model trained on 302 patient records that estimates the probability of coronary artery disease from 13 clinical measurements. Built end-to-end: from raw CSV to a live public web application.

🔗 **Live App:** [heartriskmodel.streamlit.app](https://heartriskmodel.streamlit.app)
🔗 **GitHub:** [github.com/pahulmeet-singh/Heart-Risk-AI](https://github.com/pahulmeet-singh/Heart-Risk-AI)

---

## What It Does

The user enters patient vitals in the sidebar (age, cholesterol, heart rate, ECG readings, etc.) and clicks **Generate Risk Report**. The app outputs:

- A **risk percentage** (0–100%) — probability of heart disease
- A **risk category** (Low / Moderate / High) with a clinical recommendation
- A **feature contribution chart** showing which vitals drove the prediction

---

## Project Structure

```
heart_project/
├── app.py                      ← Streamlit web app (the live interface)
├── EDA.ipynb                   ← Full EDA + model training pipeline
├── heart.csv                   ← UCI Cleveland Heart Disease dataset (303 rows)
├── heart_disease_model.pkl     ← Saved trained Logistic Regression
├── heart_scaler.pkl            ← Saved StandardScaler (fitted on training data)
├── model_columns.pkl           ← Saved feature order (alignment contract)
├── Attribute_Description.xlsx  ← Clinical feature documentation
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn joblib matplotlib reportlab

# 2. Run the app (from the project folder)
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Dataset

**UCI Cleveland Heart Disease** (Kaggle version) — 303 records, 14 columns, 1 duplicate removed.

| Split | Rows |
|-------|------|
| Training (80%) | 241 |
| Test (20%) | 61 |
| **Total (clean)** | **302** |

> ⚠️ **Important encoding note:** In this Kaggle version, `target = 0` means **heart disease present** and `target = 1` means **healthy** — the reverse of the original UCI convention. The app accounts for this by using `predict_proba[:, 0]` (probability of class 0 = disease).

---

## The 13 Input Features

| Feature | Type | Range | Clinical Meaning |
|---------|------|-------|-----------------|
| `age` | Numerical | 29–77 yrs | Patient age |
| `sex` | Binary | 0=Female, 1=Male | Biological sex |
| `cp` | Categorical | 0–3 | Chest pain type (0=Typical Angina, 3=Asymptomatic) |
| `trestbps` | Numerical | 94–200 mmHg | Resting blood pressure |
| `chol` | Numerical | 126–564 mg/dl | Serum cholesterol |
| `fbs` | Binary | 0/1 | Fasting blood sugar > 120 mg/dl |
| `restecg` | Categorical | 0–2 | Resting ECG result |
| `thalach` | Numerical | 71–202 bpm | Max heart rate during stress test |
| `exang` | Binary | 0/1 | Chest pain during exercise |
| `oldpeak` | Numerical | 0–6.2 | ST depression on ECG |
| `slope` | Categorical | 0–2 | Slope of peak ST segment |
| `ca` | Categorical | 0–3 | Major vessels visible via fluoroscopy |
| `thal` | Categorical | 0–3 | Thalassemia status |

---

## Preprocessing Pipeline (EDA.ipynb → app.py)

The same steps are applied in the notebook during training **and** in the app at prediction time. This consistency is mandatory — the scaler was fitted on training data and must receive identically-processed input.

| Step | Code | Reason |
|------|------|--------|
| Drop duplicates | `drop_duplicates()` | 1 duplicate row found |
| Cap cholesterol | 99th percentile = 406.7 mg/dl | Extreme outlier in `chol` |
| Log-transform oldpeak | `log1p(oldpeak)` | Right-skewed (skewness 1.27 → 0.4) |
| Drop original oldpeak | `df.drop('oldpeak')` | Replaced by `oldpeak_log` |
| Z-score cap ±3σ | `np.clip(col, μ-3σ, μ+3σ)` | Removes remaining extremes |
| Align columns | `df[model_columns]` | Guarantees correct feature order |
| StandardScaler | `(x − mean) / std` | Equalises feature scale for gradient descent |

---

## Algorithm: Logistic Regression

```
z = b₀ + b₁×age + b₂×sex + ... + b₁₃×oldpeak_log
P(disease) = 1 / (1 + e^(−z))    ← sigmoid function
```

The model learns 13 coefficients + 1 bias term by minimising Binary Cross-Entropy Loss. After training, `predict_proba(X)[:, 0]` returns the probability of heart disease for a new patient.

**Trained coefficients (disease risk direction):**

| Feature | Coefficient | Effect |
|---------|-------------|--------|
| `ca` (vessels blocked) | −0.833 | More vessels → higher disease risk |
| `sex` (male) | −0.829 | Male → higher disease risk |
| `thal` | −0.618 | Reversible defect → higher risk |
| `exang` | −0.505 | Exercise pain → higher risk |
| `cp` | +0.655 | Lower cp value → higher risk |
| `thalach` | +0.522 | Lower max HR → higher risk |

*(Coefficients are for P(class=1)=P(healthy); negative = associated with disease)*

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~85% |
| AUC-ROC | ~0.90 |
| Training set | 241 rows |
| Test set | 61 rows |

---

## How the Feature Contribution Chart Works

```python
contributions = -(patient_scaled_values × model.coef_[0])
```

Multiplying the patient's scaled feature values by the model's coefficients gives each feature's contribution to the prediction. Negating converts from "contribution to healthy" to "contribution to disease risk." Red bars increase risk; green bars reduce it.

---

## Tech Stack

| Tool | Role |
|------|------|
| Python 3 | Core language |
| scikit-learn | LogisticRegression, StandardScaler, metrics |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualisations |
| joblib | Model serialisation (.pkl files) |
| Streamlit | Web interface + deployment |

---

*For educational purposes only. Not for clinical use.*
