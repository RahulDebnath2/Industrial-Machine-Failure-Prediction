import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT / "data" / "synthetic_machine_data.csv"
df = pd.read_csv(data_path)

target = "Machine failure"
X, y = df.drop(columns=[target]), df[target]
categorical = ["Type"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])

pipeline = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, {"model__n_estimators":[100,200], "model__max_depth":[None,8,12]}, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
pred = best.predict(X_test)

print("Best parameters:", grid.best_params_)
print("Accuracy:", round(accuracy_score(y_test,pred),4))
print("Precision:", round(precision_score(y_test,pred,zero_division=0),4))
print("Recall:", round(recall_score(y_test,pred,zero_division=0),4))
print("F1 score:", round(f1_score(y_test,pred,zero_division=0),4))
print("Confusion matrix:\\n", confusion_matrix(y_test,pred))
print(classification_report(y_test,pred,zero_division=0))

joblib.dump(best, ROOT/"machine_failure_model.joblib")
print("Saved model to machine_failure_model.joblib")
