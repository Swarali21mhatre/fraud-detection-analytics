"""
Fraud detection model: trains a Random Forest on transactions.csv,
serialises the artefacts, and exposes a predict() function used by the API.
"""
import os, joblib, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "rf_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENC_PATH    = os.path.join(MODEL_DIR, "encoders.joblib")
META_PATH   = os.path.join(MODEL_DIR, "meta.json")

CAT_COLS = ["transaction_type", "payment_method", "location", "merchant_category"]
NUM_COLS = ["amount", "hour", "day_of_week", "is_weekend", "balance_before"]

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, encoders: dict | None = None, fit: bool = False):
    df = df.copy()

    # Amount buckets
    df["amount_log"] = np.log1p(df["amount"])
    df["is_high_amount"] = (df["amount"] > 3000).astype(int)

    # Odd-hours flag
    df["is_odd_hour"] = df["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)

    extra_num = ["amount_log", "is_high_amount", "is_odd_hour"]

    # Label-encode categoricals
    enc_out = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        if fit:
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            enc_out[col] = le
        else:
            le = encoders[col]
            # handle unseen labels
            df[col + "_enc"] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    enc_cols = [c + "_enc" for c in CAT_COLS]
    feature_cols = NUM_COLS + extra_num + enc_cols

    X = df[feature_cols].fillna(0)
    return X, (enc_out if fit else encoders), feature_cols


# ── Train ──────────────────────────────────────────────────────────────────────
def train(csv_path: str = "data/transactions.csv"):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    X, encoders, feature_cols = engineer_features(df, fit=True)
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    meta = {
        "auc":       round(float(roc_auc_score(y_test, y_proba)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm,
        "feature_cols": feature_cols,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "fraud_rate_train": round(float(y_train.mean()), 4),
    }

    joblib.dump(model,    MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(encoders, ENC_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("=== Model trained ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {meta['auc']}")
    return model, scaler, encoders, meta


# ── Load (lazy, cached) ───────────────────────────────────────────────────────
_cache: dict = {}

def _load():
    if _cache:
        return _cache
    if not os.path.exists(MODEL_PATH):
        train()
    _cache["model"]    = joblib.load(MODEL_PATH)
    _cache["scaler"]   = joblib.load(SCALER_PATH)
    _cache["encoders"] = joblib.load(ENC_PATH)
    with open(META_PATH) as f:
        _cache["meta"] = json.load(f)
    return _cache


# ── Public predict API ────────────────────────────────────────────────────────
def predict_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Return fraud probability [0,1] for each row."""
    arts = _load()
    df = df.copy()
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "hour" not in df.columns:
        df["hour"]        = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    X, _, _ = engineer_features(df, encoders=arts["encoders"], fit=False)
    X_s = arts["scaler"].transform(X)
    return arts["model"].predict_proba(X_s)[:, 1]


def risk_score(prob: float) -> int:
    """Convert probability [0,1] → integer risk score [0,100]."""
    return int(round(prob * 100))


if __name__ == "__main__":
    train()
