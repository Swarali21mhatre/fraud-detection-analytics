"""
Flask REST API for Fraud Detection Analytics.
All heavy lifting (ML, aggregations) happens in analytics.py.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import analytics


# ─────────────────────────────────────────────────────────────────────────────
# Flask Application
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
CORS(app)


# ─────────────────────────────────────────────────────────────────────────────
# Load data and ML model when the application starts
# This is important for cloud deployment with Gunicorn.
# ─────────────────────────────────────────────────────────────────────────────

analytics.load()


# ─────────────────────────────────────────────────────────────────────────────
# Serve Frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Fraud Detection API is running"
    })


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard KPIs
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/dashboard")
def dashboard():
    return jsonify(
        analytics.get_dashboard_kpis()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/model-metrics")
def model_metrics():
    return jsonify(
        analytics.get_model_metrics()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chart Data
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/charts/fraud-over-time")
def fraud_over_time():
    return jsonify(
        analytics.fraud_over_time()
    )


@app.route("/api/charts/by-type")
def by_type():
    return jsonify(
        analytics.fraud_by_type()
    )


@app.route("/api/charts/by-payment-method")
def by_payment_method():
    return jsonify(
        analytics.fraud_by_payment_method()
    )


@app.route("/api/charts/by-location")
def by_location():
    return jsonify(
        analytics.fraud_by_location()
    )


@app.route("/api/charts/amount-distribution")
def amount_distribution():
    return jsonify(
        analytics.amount_distribution()
    )


@app.route("/api/charts/hourly-pattern")
def hourly_pattern():
    return jsonify(
        analytics.hourly_pattern()
    )


@app.route("/api/charts/risk-score-distribution")
def risk_score_dist():
    return jsonify(
        analytics.risk_score_distribution()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transactions
# Pagination, Search, Filtering and Sorting
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/transactions")
def transactions():

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 25))

    search = request.args.get(
        "search", ""
    ).strip()

    tx_type = request.args.get(
        "type", ""
    ).strip()

    method = request.args.get(
        "method", ""
    ).strip()

    location = request.args.get(
        "location", ""
    ).strip()

    status = request.args.get(
        "status", ""
    ).strip()

    sort_by = request.args.get(
        "sort", "timestamp"
    )

    order = request.args.get(
        "order", "desc"
    )

    min_amt = request.args.get(
        "min_amount",
        type=float
    )

    max_amt = request.args.get(
        "max_amount",
        type=float
    )

    min_risk = request.args.get(
        "min_risk",
        type=int
    )

    max_risk = request.args.get(
        "max_risk",
        type=int
    )

    result = analytics.get_transactions(
        page=page,
        per_page=per_page,
        search=search,
        tx_type=tx_type,
        method=method,
        location=location,
        status=status,
        sort_by=sort_by,
        order=order,
        min_amount=min_amt,
        max_amount=max_amt,
        min_risk=min_risk,
        max_risk=max_risk,
    )

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# Single Transaction Details
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/transactions/<txn_id>")
def transaction_detail(txn_id):

    row = analytics.get_transaction_detail(txn_id)

    if row is None:
        return jsonify({
            "error": "Transaction not found"
        }), 404

    return jsonify(row)


# ─────────────────────────────────────────────────────────────────────────────
# Filter Options
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/filter-options")
def filter_options():
    return jsonify(
        analytics.get_filter_options()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Application Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Local development server
    # Render/Gunicorn will not use this block.
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )