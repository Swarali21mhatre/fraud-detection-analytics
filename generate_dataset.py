"""
Generate a realistic synthetic fraud detection dataset.
Run once to produce data/transactions.csv
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

N_LEGIT = 9500
N_FRAUD = 500
N = N_LEGIT + N_FRAUD

os.makedirs("data", exist_ok=True)

# ── Categorical pools ──────────────────────────────────────────────────────────
TRANSACTION_TYPES = ["purchase", "transfer", "withdrawal", "payment", "refund"]
PAYMENT_METHODS   = ["credit_card", "debit_card", "bank_transfer", "digital_wallet", "crypto"]
LOCATIONS         = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "London",
    "Frankfurt", "Singapore", "Tokyo", "Dubai", "Sydney",
]
MERCHANT_CATS     = [
    "retail", "food_beverage", "travel", "entertainment",
    "healthcare", "utilities", "electronics", "luxury_goods",
]

# ── Helper: random timestamps over the past 6 months ──────────────────────────
start = datetime(2024, 1, 1)
end   = datetime(2024, 6, 30, 23, 59, 59)
span  = int((end - start).total_seconds())

def rand_ts(n, skew_hour=None):
    secs = rng.integers(0, span, size=n)
    if skew_hour is not None:
        # fraud tends to happen at odd hours (0-5)
        odd = rng.integers(0, 6 * 3600, size=n)
        mask = rng.random(n) < 0.6
        secs = np.where(mask, odd, secs)
    return [start + timedelta(seconds=int(s)) for s in secs]

# ── Legitimate transactions ────────────────────────────────────────────────────
legit_amounts = np.round(
    np.concatenate([
        rng.exponential(scale=80,  size=int(N_LEGIT * 0.6)),
        rng.uniform(100, 2000,     size=int(N_LEGIT * 0.3)),
        rng.uniform(2000, 8000,    size=int(N_LEGIT * 0.1)),
    ]), 2
)
rng.shuffle(legit_amounts)

legit = pd.DataFrame({
    "transaction_id":   [f"TXN{i:06d}" for i in range(1, N_LEGIT + 1)],
    "timestamp":        rand_ts(N_LEGIT),
    "amount":           legit_amounts,
    "transaction_type": rng.choice(TRANSACTION_TYPES, size=N_LEGIT, p=[0.40, 0.25, 0.15, 0.15, 0.05]),
    "payment_method":   rng.choice(PAYMENT_METHODS,   size=N_LEGIT, p=[0.35, 0.30, 0.20, 0.12, 0.03]),
    "location":         rng.choice(LOCATIONS[:10],    size=N_LEGIT),   # mostly domestic
    "merchant_category":rng.choice(MERCHANT_CATS,     size=N_LEGIT),
    "is_fraud":         0,
})

# ── Fraudulent transactions ────────────────────────────────────────────────────
fraud_amounts = np.round(
    np.concatenate([
        rng.uniform(1000, 5000,  size=int(N_FRAUD * 0.5)),
        rng.uniform(5000, 20000, size=int(N_FRAUD * 0.3)),
        rng.uniform(0.5,  50,    size=int(N_FRAUD * 0.2)),   # micro-probe transactions
    ]), 2
)
rng.shuffle(fraud_amounts)

fraud = pd.DataFrame({
    "transaction_id":   [f"TXN{i:06d}" for i in range(N_LEGIT + 1, N + 1)],
    "timestamp":        rand_ts(N_FRAUD, skew_hour=True),
    "amount":           fraud_amounts,
    "transaction_type": rng.choice(TRANSACTION_TYPES, size=N_FRAUD, p=[0.20, 0.40, 0.25, 0.10, 0.05]),
    "payment_method":   rng.choice(PAYMENT_METHODS,   size=N_FRAUD, p=[0.30, 0.15, 0.15, 0.15, 0.25]),
    "location":         rng.choice(LOCATIONS,          size=N_FRAUD),  # includes international
    "merchant_category":rng.choice(MERCHANT_CATS,      size=N_FRAUD, p=[0.10, 0.05, 0.25, 0.15, 0.05, 0.05, 0.25, 0.10]),
    "is_fraud":         1,
})

# ── Combine, shuffle, add account IDs ─────────────────────────────────────────
df = pd.concat([legit, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# 500 unique accounts
df["account_id"] = [f"ACC{rng.integers(1000, 1500):04d}" for _ in range(len(df))]
# balance snapshot (approx)
df["balance_before"] = np.round(rng.uniform(500, 50000, size=len(df)), 2)
df["balance_after"]  = np.round(
    np.where(df["is_fraud"] == 1,
             df["balance_before"] - df["amount"],
             df["balance_before"] - df["amount"] * rng.uniform(0.5, 1.0, size=len(df))), 2
)

# ── Engineered features used by model later ───────────────────────────────────
df["hour"]        = df["timestamp"].apply(lambda t: t.hour)
df["day_of_week"] = df["timestamp"].apply(lambda t: t.weekday())
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

df.to_csv("data/transactions.csv", index=False)
print(f"Dataset saved: {len(df)} rows  |  fraud={df['is_fraud'].sum()}  |  legit={N_LEGIT}")
