import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "machine_failure_model.joblib"
if not model_path.exists():
    raise SystemExit("Model file not found. Run src/train_model.py first.")

model = joblib.load(model_path)
sample = pd.DataFrame([{
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 120.0
}])
prediction = int(model.predict(sample)[0])
print("Predicted machine failure:", "Yes" if prediction == 1 else "No")
if hasattr(model, "predict_proba"):
    print("Failure probability:", round(float(model.predict_proba(sample)[0][1]), 3))
