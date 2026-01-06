from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier

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
# LOAD DATA (untuk dropdown UI + training)
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

# Validasi kolom
missing_cols = [c for c in FEATURE_ORDER + [TARGET_COL] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Kolom berikut tidak ditemukan di heart.csv: {missing_cols}")

# =========================
# TRAIN MODEL (tanpa .pkl)
# =========================
# Pastikan numerik
X = df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").to_numpy()
y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int).to_numpy()

# scale_pos_weight = neg/pos (sesuai TA-14 jika imbalanced)
neg = int((y == 0).sum())
pos = int((y == 1).sum())
if pos == 0:
    raise ValueError("Tidak ada kelas positif (1) pada dataset. Periksa label target.")

scale_pos_weight = neg / pos
is_imbalanced = (neg / max(pos, 1)) >= 1.5  # heuristik opsional

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight if is_imbalanced else 1.0
)

# XGBoost bisa handle NaN otomatis
model.fit(X, y)

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

        # Ambil input user sesuai urutan fitur
        for col in FEATURE_ORDER:
            val = request.form.get(col, "").strip()
            if val == "":
                raise ValueError(f"Kolom '{col}' belum diisi.")

            try:
                features.append(float(val))
            except ValueError:
                raise ValueError(f"Nilai '{col}' harus berupa angka.")

            filled[col] = val

        X_input = np.array([features], dtype=float)

        # Prediksi
        pred = int(model.predict(X_input)[0])
        proba = model.predict_proba(X_input)[0]
        confidence = float(np.max(proba) * 100)

        result_text = (
            "💔 Pasien <b>TERINDIKASI</b> penyakit jantung (target = 1)"
            if pred == 1 else
            "💖 Pasien <b>TIDAK</b> terindikasi penyakit jantung (target = 0)"
        )

        proba_text = (
            f"Probabilitas → "
            f"P(0) = {proba[0]:.4f}, "
            f"P(1) = {proba[1]:.4f} | "
            f"Keyakinan model: {confidence:.2f}%"
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

# =========================
# RUN LOCAL
# =========================
if __name__ == "__main__":
    app.run(debug=True)
