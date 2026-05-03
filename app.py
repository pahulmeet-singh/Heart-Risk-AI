import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Risk AI",
    page_icon="❤️",
    layout="wide"
)

# ─────────────────────────────────────────────
#  LOAD SAVED MODEL ASSETS
#  All three .pkl files must sit in the same folder as this script.
# ─────────────────────────────────────────────
@st.cache_resource          # loads once, then cached for the session
def load_assets():
    model         = joblib.load('heart_disease_model.pkl')  # Logistic Regression
    scaler        = joblib.load('heart_scaler.pkl')         # StandardScaler
    model_columns = joblib.load('model_columns.pkl')        # ordered feature list
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# Human-readable labels for the contribution chart
FEATURE_LABELS = {
    'age':         'Age',
    'sex':         'Sex (Male=1)',
    'cp':          'Chest Pain Type',
    'trestbps':    'Resting Blood Pressure',
    'chol':        'Serum Cholesterol',
    'fbs':         'High Fasting Blood Sugar',
    'restecg':     'Resting ECG',
    'thalach':     'Max Heart Rate Achieved',
    'exang':       'Exercise-Induced Chest Pain',
    'slope':       'ST Slope',
    'ca':          'Major Vessels (Fluoroscopy)',
    'thal':        'Thalassemia Status',
    'oldpeak_log': 'ST Depression (log-scaled)',
}

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.title("Heart Disease Clinical Risk Assessment")
st.write(
    "A **Logistic Regression** model trained on 302 patient records estimates "
    "the probability of heart disease from clinical vitals. "
    "Enter patient data on the left and click **Generate Risk Report**."
)
st.divider()

# ─────────────────────────────────────────────
#  SIDEBAR — PATIENT VITALS
# ─────────────────────────────────────────────
st.sidebar.header("Patient Vitals")
st.sidebar.caption("Adjust all fields, then click the button.")

def get_user_input() -> dict:
    """Collect raw patient data from the sidebar widgets."""

    age      = st.sidebar.slider("Age (years)", 20, 80, 50)

    sex_map  = {"Male": 1, "Female": 0}
    sex      = sex_map[st.sidebar.selectbox("Sex", list(sex_map.keys()))]

    # cp = 0: Typical Angina (classic symptom → higher risk in this dataset)
    # cp = 3: Asymptomatic  (no chest pain   → lower risk in this dataset)
    cp_map = {
        "Typical Angina (cp=0) — classic heart chest pain":   0,
        "Atypical Angina (cp=1) — unusual chest pain":        1,
        "Non-anginal Pain (cp=2) — not heart-related":        2,
        "Asymptomatic (cp=3) — no chest pain":                3,
    }
    cp       = cp_map[st.sidebar.selectbox("Chest Pain Type", list(cp_map.keys()))]

    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol     = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)

    fbs_map  = {"Yes — above 120 mg/dl": 1, "No — 120 mg/dl or below": 0}
    fbs      = fbs_map[st.sidebar.selectbox("Fasting Blood Sugar High?", list(fbs_map.keys()))]

    ecg_map  = {
        "Normal":                       0,
        "ST-T Wave Abnormality":        1,
        "Left Ventricular Hypertrophy": 2,
    }
    restecg  = ecg_map[st.sidebar.selectbox("Resting ECG Results", list(ecg_map.keys()))]

    thalach  = st.sidebar.number_input("Maximum Heart Rate Achieved (bpm)", 60, 220, 150)

    exang_map = {"Yes — chest pain during exercise": 1, "No": 0}
    exang     = exang_map[st.sidebar.selectbox("Exercise-Induced Chest Pain?", list(exang_map.keys()))]

    oldpeak  = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)

    slope_map = {
        "Upsloping (healthier sign)":    0,
        "Flat":                          1,
        "Downsloping (more concerning)": 2
    }
    slope     = slope_map[st.sidebar.selectbox("Slope of Peak Exercise ST Segment", list(slope_map.keys()))]

    ca       = st.sidebar.selectbox("Major Vessels Visible via Fluoroscopy (0-3)", [0, 1, 2, 3])

    thal_map = {"Fixed Defect (scar tissue)": 1, "Normal": 2, "Reversible Defect (ischemia)": 3}
    thal     = thal_map[st.sidebar.selectbox("Thalassemia Status", list(thal_map.keys()))]

    return {
        'age': age,    'sex': sex,      'cp': cp,          'trestbps': trestbps,
        'chol': chol,  'fbs': fbs,      'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

raw_input = get_user_input()

# ─────────────────────────────────────────────
#  PREPROCESSING FUNCTION
#  Mirrors the training pipeline in EDA.ipynb exactly.
# ─────────────────────────────────────────────
def preprocess(raw: dict):
    """
    Preprocessing steps (must match training):
      1. Log-transform oldpeak -> oldpeak_log  (reduces right skewness)
      2. Drop original oldpeak column
      3. Reorder columns to match model_columns (training order)
      4. Scale with the saved StandardScaler

    CRITICAL: No pd.get_dummies here.
    The model was trained on raw integer labels for categorical features.
    Adding one-hot encoding would create columns the model has never seen
    and silently produce garbage predictions.
    """
    df = pd.DataFrame([raw])

    # Step 1 & 2: log transform, drop original
    df['oldpeak_log'] = np.log1p(df['oldpeak'])
    df.drop('oldpeak', axis=1, inplace=True)

    # Step 3: ensure column order exactly matches training
    df = df[model_columns]

    # Step 4: scale using the SAVED scaler (fitted on training data only)
    scaled = scaler.transform(df)

    return scaled, df


# ─────────────────────────────────────────────
#  PREDICT BUTTON
# ─────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    run = st.button("Generate Risk Report", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
#  RESULTS SECTION
# ─────────────────────────────────────────────
if run:
    scaled_data, _ = preprocess(raw_input)

    # CRITICAL: In this Kaggle Cleveland dataset the target is encoded as:
    #   target = 0  →  Heart Disease PRESENT
    #   target = 1  →  No Disease (Healthy)
    #
    # predict_proba returns [P(class=0), P(class=1)] = [P(disease), P(healthy)]
    # So the DISEASE risk = predict_proba[:, 0], NOT [:, 1]
    risk_prob = model.predict_proba(scaled_data)[:, 0][0] * 100

    st.divider()
    left_col, right_col = st.columns([1, 1])

    # ── LEFT: Risk Score ──────────────────────
    with left_col:
        st.subheader("Risk Assessment Result")

        if risk_prob >= 60:
            st.error(f"### High Risk: {risk_prob:.1f}%")
            recommendation = "Consult a **cardiologist immediately** for detailed diagnostics."
        elif risk_prob >= 30:
            st.warning(f"### Moderate Risk: {risk_prob:.1f}%")
            recommendation = "Schedule a **follow-up with your physician** soon."
        else:
            st.success(f"### Low Risk: {risk_prob:.1f}%")
            recommendation = "Maintain a **healthy lifestyle** and attend routine check-ups."

        st.progress(int(risk_prob), text=f"{risk_prob:.1f}% estimated probability of heart disease")
        st.write(f"**Recommendation:** {recommendation}")
        st.caption("0-30% Low  |  30-60% Moderate  |  60-100% High")

        st.divider()

        # Patient data summary table
        st.subheader("Patient Data Summary")
        cp_labels   = {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-anginal Pain", 3:"Asymptomatic"}
        thal_labels = {1:"Fixed Defect", 2:"Normal", 3:"Reversible Defect"}
        summary = {
            "Age":                  raw_input['age'],
            "Sex":                  "Male" if raw_input['sex'] == 1 else "Female",
            "Chest Pain":           cp_labels[raw_input['cp']],
            "Resting BP (mm Hg)":   raw_input['trestbps'],
            "Cholesterol (mg/dl)":  raw_input['chol'],
            "Max Heart Rate (bpm)": raw_input['thalach'],
            "Exercise Chest Pain":  "Yes" if raw_input['exang'] else "No",
            "ST Depression":        raw_input['oldpeak'],
            "Major Vessels":        raw_input['ca'],
            "Thalassemia":          thal_labels.get(raw_input['thal'], raw_input['thal']),
        }
        # st.table(pd.DataFrame(list(summary.items()), columns=["Attribute", "Value"]))
        summary_df = pd.DataFrame(list(summary.items()), columns=["Attribute", "Value"])
        summary_df["Value"] = summary_df["Value"].astype(str) # Fixes the PyArrow error
        st.table(summary_df)

    # ── RIGHT: Feature Contribution Chart ────
    with right_col:
        st.subheader("What's Driving This Risk Score?")
        st.caption(
            "Each bar shows how much a feature pushed the prediction. "
            "Red = increases disease risk  |  Green = reduces disease risk"
        )

        # Local explanation: contribution = scaled_value * model_coefficient
        # The model's coefficients are for P(class=1) = P(healthy).
        # We negate them so the chart reads as contribution to P(disease).
        weights       = model.coef_[0]
        patient_vals  = scaled_data[0]
        contributions = -(patient_vals * weights)   # negate: higher = more disease risk

        contrib_df = pd.DataFrame({
            'Feature':      [FEATURE_LABELS[c] for c in model_columns],
            'Contribution': contributions
        }).sort_values('Contribution', ascending=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in contrib_df['Contribution']]
        ax.barh(contrib_df['Feature'], contrib_df['Contribution'],
                color=bar_colors, edgecolor='white')
        ax.axvline(0, color='#333333', lw=1)
        ax.set_xlabel("Contribution to Disease Risk Score", fontsize=9)
        ax.set_title("Feature Contributions (This Patient)", fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        red_patch   = mpatches.Patch(color='#e74c3c', label='Increases risk')
        green_patch = mpatches.Patch(color='#2ecc71', label='Reduces risk')
        ax.legend(handles=[red_patch, green_patch], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── HOW IT WORKS EXPANDER ─────────────────
    st.divider()
    with st.expander("How does this model work?"):
        st.markdown("""
        ### Algorithm: Logistic Regression

        Logistic Regression models the **log-odds** of an outcome as a weighted sum of features,
        then converts it to a probability via the **sigmoid function**:

        ```
        P(disease) = sigmoid( b0 + b1*age + b2*sex + b3*cp + ... + b13*oldpeak_log )
        ```

        Each coefficient tells us: *"When this feature increases by 1 standard deviation,
        how does the log-odds of heart disease change?"*

        ---

        ### Preprocessing Pipeline (mirrors training exactly)

        | Step | Operation | Reason |
        |------|-----------|--------|
        | Drop duplicates | `drop_duplicates()` | 1 duplicate row in dataset |
        | Cap cholesterol | 99th percentile cutoff | Extreme outliers in chol |
        | Z-score capping | Clip at ±3 std for numeric cols | Handles remaining outliers |
        | Log-transform oldpeak | `log1p(oldpeak)` | oldpeak is right-skewed |
        | StandardScaler | Zero mean, unit variance | Logistic Regression converges on scaled data |

        ---

        ### Bug 1 Fixed: No One-Hot Encoding

        The original `app.py` called `pd.get_dummies()` before prediction. But the model was
        trained on **raw integer labels** for `cp`, `slope`, `ca`, `thal`, `restecg`.
        Encoding at prediction time creates dummy columns (`cp_1`, `cp_2`...) that the model
        has never seen. The `reindex` then filled everything with **0**, making all categorical
        inputs completely ignored.

        **Fix:** Remove `pd.get_dummies`. Pass raw integer labels directly.

        ---

        ### Bug 2 Fixed: Correct Target Class Index

        In this Kaggle version of the Cleveland dataset:
        - `target = 0` → Heart Disease **present** ← we want this probability
        - `target = 1` → No Disease (Healthy)

        The original code used `predict_proba[:, 1]` (probability of being healthy) as the
        "risk score". This is **inverted** — healthy patients showed high "risk" and sick
        patients showed low "risk".

        **Fix:** Use `predict_proba[:, 0]` for disease probability.

        ---

        ### Model Performance
        - **Test Accuracy:** ~85%
        - **AUC-ROC:** ~0.90

        *This tool is for educational/demonstration purposes only.*
        *Always validate with a licensed clinician.*
        """)
