from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart.csv")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

TARGET_COL = "target"

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

CATEGORICAL_COLS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

# =========================
# SIMPLE LIGHTWEIGHT MODEL
# =========================
class SimpleHeartModel:
    def __init__(self, df):
        self.means = df[FEATURE_ORDER].mean().to_numpy()
        self.stds = df[FEATURE_ORDER].std().replace(0, 1).to_numpy()

    def predict_proba(self, X):
        X_norm = (X - self.means) / self.stds
        score = X_norm.mean(axis=1)
        prob_1 = 1 / (1 + np.exp(-score))
        prob_0 = 1 - prob_1
        return np.vstack([prob_0, prob_1]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

model = SimpleHeartModel(df)

# =========================
# DROPDOWN DATA
# =========================
def build_dropdown_values(df: pd.DataFrame) -> dict:
    dropdowns = {}
    for col in CATEGORICAL_COLS:
        vals = df[col].dropna().unique().tolist()
        try:
            vals = sorted(vals, key=lambda v: float(v))
        except Exception:
            vals = sorted(vals)
        dropdowns[col] = vals
    return dropdowns

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template(
        "index.html",
        data=build_dropdown_values(df),
        feature_cols=FEATURE_ORDER,
        filled=None,
        labels=None,
        values=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    dropdown_data = build_dropdown_values(df)

    try:
        filled = {}
        features = []

        for col in FEATURE_ORDER:
            val = request.form.get(col, "").strip()
            if val == "":
                raise ValueError(f"Kolom '{col}' belum diisi.")
            features.append(float(val))
            filled[col] = val

        X_input = np.array([features], dtype=float)

        pred = int(model.predict(X_input)[0])
        proba = model.predict_proba(X_input)[0]
        confidence = float(np.max(proba) * 100)

        result_text = (
            "💔 Pasien <b>TERINDIKASI</b> penyakit jantung (target = 1)"
            if pred == 1 else
            "💖 Pasien <b>TIDAK</b> terindikasi penyakit jantung (target = 0)"
        )

        proba_text = (
            f"P(0) = {proba[0]:.4f}, "
            f"P(1) = {proba[1]:.4f} | "
            f"Keyakinan: {confidence:.2f}%"
        )

        return render_template(
            "index.html",
            prediction_text=result_text,
            proba_text=proba_text,
            data=dropdown_data,
            feature_cols=FEATURE_ORDER,
            filled=filled,
            labels=["Tidak (0)", "Ya (1)"],
            values=[float(proba[0]), float(proba[1])]
        )

    except Exception as e:
        return render_template(
            "index.html",
            error_text=f"Terjadi error: {e}",
            data=dropdown_data,
            feature_cols=FEATURE_ORDER,
            filled=None,
            labels=None,
            values=None
        )

if __name__ == "__main__":
    app.run(debug=True)
