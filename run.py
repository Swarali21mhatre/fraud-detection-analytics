"""
One-click setup and launch for Fraud Detection Analytics.
Run:  python run.py
"""
import subprocess, sys, os

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])

print("=" * 60)
print("  Fraud Detection & Risk Analytics — Setup")
print("=" * 60)

# 1. Install dependencies
print("\n[1/3] Installing dependencies…")
pip("install", "-q", "-r", "requirements.txt")

# 2. Generate dataset
if not os.path.exists("data/transactions.csv"):
    print("\n[2/3] Generating transaction dataset…")
    subprocess.check_call([sys.executable, "generate_dataset.py"])
else:
    print("\n[2/3] Dataset already exists — skipping generation.")

# 3. Train model
if not os.path.exists("model/rf_model.joblib"):
    print("\n[3/3] Training Random Forest model…")
    subprocess.check_call([sys.executable, "-c",
        "from fraud_model import train; train()"])
else:
    print("\n[3/3] Model already trained — skipping.")

print("\n✓ Setup complete.")
print("  Starting Flask server on  http://localhost:5000")
print("  Press Ctrl+C to stop.\n")

os.execv(sys.executable, [sys.executable, "app.py"])
