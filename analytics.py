"""
Analytics layer: loads transaction data, runs ML predictions,
and computes all aggregations consumed by the Flask API.
"""
import json, os
import numpy as np
import pandas as pd
import fraud_model

_df: pd.DataFrame | None = None   # full enriched dataframe (singleton)


# ── Load & enrich ──────────────────────────────────────────────────────────────
def load(csv_path: str = "data/transactions.csv") -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Derived time features
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["date"]        = df["timestamp"].dt.date
    df["week"]        = df["timestamp"].dt.to_period("W").apply(lambda r: str(r.start_time.date()))
    df["month"]       = df["timestamp"].dt.to_period("M").astype(str)

    # ML risk scores
    probs = fraud_model.predict_dataframe(df)
    df["fraud_probability"] = np.round(probs, 4)
    df["risk_score"]        = (probs * 100).round().astype(int)

    # Derived label (model prediction at threshold 0.5)
    df["predicted_fraud"] = (probs >= 0.5).astype(int)

    # Risk tier
    def tier(s):
        if s >= 75: return "Critical"
        if s >= 50: return "High"
        if s >= 25: return "Medium"
        return "Low"
    df["risk_tier"] = df["risk_score"].apply(tier)

    # Normalise timestamp for JSON
    df["timestamp_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    _df = df
    return _df


def _get() -> pd.DataFrame:
    if _df is None:
        load()
    return _df


# ── KPIs ───────────────────────────────────────────────────────────────────────
def get_dashboard_kpis() -> dict:
    df = _get()
    total         = len(df)
    fraud_rows    = df[df["is_fraud"] == 1]
    n_fraud       = len(fraud_rows)
    fraud_amount  = float(fraud_rows["amount"].sum())
    total_amount  = float(df["amount"].sum())
    fraud_rate    = round(n_fraud / total * 100, 2)
    avg_risk      = round(float(df["risk_score"].mean()), 1)
    high_risk     = int((df["risk_score"] >= 75).sum())

    # Month-over-month delta (last 2 months)
    months = sorted(df["month"].unique())
    if len(months) >= 2:
        m1 = df[df["month"] == months[-2]]
        m2 = df[df["month"] == months[-1]]
        mom_delta = round(
            (len(m2[m2["is_fraud"] == 1]) / max(len(m2), 1) -
             len(m1[m1["is_fraud"] == 1]) / max(len(m1), 1)) * 100, 2
        )
    else:
        mom_delta = 0.0

    return {
        "total_transactions": total,
        "fraud_count":        n_fraud,
        "legit_count":        total - n_fraud,
        "fraud_rate_pct":     fraud_rate,
        "fraud_amount":       round(fraud_amount, 2),
        "total_amount":       round(total_amount, 2),
        "fraud_amount_pct":   round(fraud_amount / max(total_amount, 1) * 100, 2),
        "avg_risk_score":     avg_risk,
        "high_risk_count":    high_risk,
        "mom_fraud_delta":    mom_delta,
    }


# ── Model metrics ──────────────────────────────────────────────────────────────
def get_model_metrics() -> dict:
    arts = fraud_model._load()
    return arts["meta"]


# ── Charts ─────────────────────────────────────────────────────────────────────
def fraud_over_time() -> dict:
    df = _get()
    grp = df.groupby(["week", "is_fraud"]).size().unstack(fill_value=0).reset_index()
    grp.columns.name = None
    if 0 not in grp: grp[0] = 0
    if 1 not in grp: grp[1] = 0
    grp = grp.rename(columns={0: "legit", 1: "fraud"})
    grp = grp.sort_values("week")
    return {
        "labels":  grp["week"].tolist(),
        "legit":   grp["legit"].tolist(),
        "fraud":   grp["fraud"].tolist(),
    }


def fraud_by_type() -> dict:
    df = _get()
    grp = df.groupby("transaction_type").agg(
        total=("is_fraud", "count"),
        fraud=("is_fraud", "sum"),
    ).reset_index()
    grp["fraud_rate"] = (grp["fraud"] / grp["total"] * 100).round(2)
    return grp.sort_values("fraud_rate", ascending=False).to_dict(orient="records")


def fraud_by_payment_method() -> dict:
    df = _get()
    grp = df.groupby("payment_method").agg(
        total=("is_fraud", "count"),
        fraud=("is_fraud", "sum"),
        amount=("amount", "sum"),
    ).reset_index()
    grp["fraud_rate"] = (grp["fraud"] / grp["total"] * 100).round(2)
    return grp.sort_values("fraud_rate", ascending=False).to_dict(orient="records")


def fraud_by_location() -> dict:
    df = _get()
    grp = df.groupby("location").agg(
        total=("is_fraud", "count"),
        fraud=("is_fraud", "sum"),
        amount=("amount", "sum"),
    ).reset_index()
    grp["fraud_rate"] = (grp["fraud"] / grp["total"] * 100).round(2)
    return grp.sort_values("fraud", ascending=False).to_dict(orient="records")


def amount_distribution() -> dict:
    df = _get()
    bins = [0, 50, 200, 500, 1000, 2000, 5000, 10000, float("inf")]
    labels = ["$0-50","$50-200","$200-500","$500-1K","$1K-2K","$2K-5K","$5K-10K","$10K+"]
    df["amt_bin"] = pd.cut(df["amount"], bins=bins, labels=labels, right=False)
    grp = df.groupby(["amt_bin", "is_fraud"]).size().unstack(fill_value=0).reset_index()
    grp.columns.name = None
    if 0 not in grp: grp[0] = 0
    if 1 not in grp: grp[1] = 0
    grp = grp.rename(columns={0: "legit", 1: "fraud"})
    return {
        "labels": labels,
        "legit":  grp["legit"].tolist(),
        "fraud":  grp["fraud"].tolist(),
    }


def hourly_pattern() -> dict:
    df = _get()
    grp = df.groupby(["hour", "is_fraud"]).size().unstack(fill_value=0).reset_index()
    grp.columns.name = None
    if 0 not in grp: grp[0] = 0
    if 1 not in grp: grp[1] = 0
    grp = grp.rename(columns={0: "legit", 1: "fraud"})
    grp = grp.sort_values("hour")
    return {
        "hours": grp["hour"].tolist(),
        "legit": grp["legit"].tolist(),
        "fraud": grp["fraud"].tolist(),
    }


def risk_score_distribution() -> dict:
    df = _get()
    bins   = list(range(0, 101, 10))
    labels = [f"{b}-{b+9}" for b in range(0, 100, 10)]
    df["rs_bin"] = pd.cut(df["risk_score"], bins=bins, labels=labels, right=False, include_lowest=True)
    counts = df.groupby("rs_bin").size().reindex(labels, fill_value=0)
    fraud_in_bin = df[df["is_fraud"]==1].groupby("rs_bin").size().reindex(labels, fill_value=0)
    return {
        "labels":      labels,
        "total":       counts.tolist(),
        "fraud":       fraud_in_bin.tolist(),
    }


# ── Transactions list ──────────────────────────────────────────────────────────
def get_transactions(
    page=1, per_page=25,
    search="", tx_type="", method="", location="", status="",
    sort_by="timestamp", order="desc",
    min_amount=None, max_amount=None,
    min_risk=None, max_risk=None,
) -> dict:
    df = _get().copy()

    # ── Filters ──
    if search:
        mask = (
            df["transaction_id"].str.contains(search, case=False, na=False) |
            df["account_id"].str.contains(search, case=False, na=False) |
            df["location"].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    if tx_type:
        df = df[df["transaction_type"] == tx_type]
    if method:
        df = df[df["payment_method"] == method]
    if location:
        df = df[df["location"] == location]
    if status == "fraud":
        df = df[df["is_fraud"] == 1]
    elif status == "legit":
        df = df[df["is_fraud"] == 0]
    if min_amount is not None:
        df = df[df["amount"] >= min_amount]
    if max_amount is not None:
        df = df[df["amount"] <= max_amount]
    if min_risk is not None:
        df = df[df["risk_score"] >= min_risk]
    if max_risk is not None:
        df = df[df["risk_score"] <= max_risk]

    # ── Sort ──
    valid_sort = {"timestamp", "amount", "risk_score", "transaction_id"}
    if sort_by not in valid_sort:
        sort_by = "timestamp"
    ascending = order == "asc"
    if sort_by == "timestamp":
        df = df.sort_values("timestamp", ascending=ascending)
    else:
        df = df.sort_values(sort_by, ascending=ascending)

    total = len(df)
    start = (page - 1) * per_page
    page_df = df.iloc[start : start + per_page]

    cols = [
        "transaction_id", "account_id", "timestamp_str", "amount",
        "transaction_type", "payment_method", "location", "merchant_category",
        "risk_score", "risk_tier", "fraud_probability", "is_fraud",
        "balance_before", "balance_after",
    ]
    records = page_df[cols].rename(columns={"timestamp_str": "timestamp"}).to_dict(orient="records")

    return {
        "transactions": records,
        "total":        total,
        "page":         page,
        "per_page":     per_page,
        "total_pages":  (total + per_page - 1) // per_page,
    }


# ── Single transaction ─────────────────────────────────────────────────────────
def get_transaction_detail(txn_id: str) -> dict | None:
    df = _get()
    rows = df[df["transaction_id"] == txn_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "transaction_id":   row["transaction_id"],
        "account_id":       row["account_id"],
        "timestamp":        row["timestamp_str"],
        "amount":           float(row["amount"]),
        "transaction_type": row["transaction_type"],
        "payment_method":   row["payment_method"],
        "location":         row["location"],
        "merchant_category":row["merchant_category"],
        "risk_score":       int(row["risk_score"]),
        "risk_tier":        row["risk_tier"],
        "fraud_probability":float(row["fraud_probability"]),
        "is_fraud":         int(row["is_fraud"]),
        "predicted_fraud":  int(row["predicted_fraud"]),
        "balance_before":   float(row["balance_before"]),
        "balance_after":    float(row["balance_after"]),
        "hour":             int(row["hour"]),
        "day_of_week":      int(row["day_of_week"]),
        "is_weekend":       int(row["is_weekend"]),
    }


# ── Filter options ─────────────────────────────────────────────────────────────
def get_filter_options() -> dict:
    df = _get()
    return {
        "transaction_types":  sorted(df["transaction_type"].unique().tolist()),
        "payment_methods":    sorted(df["payment_method"].unique().tolist()),
        "locations":          sorted(df["location"].unique().tolist()),
        "risk_tiers":         ["Low", "Medium", "High", "Critical"],
    }
