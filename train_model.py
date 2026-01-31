import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1) Load dataset
# ---------------------------
df = pd.read_csv("loan_prediction.csv")  # keep in same folder

# Drop Loan_ID if exists
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# ---------------------------
# 2) Separate X and y
# ---------------------------
target_col = "Loan_Status"
X = df.drop(columns=[target_col])
y = df[target_col].map({"N": 0, "Y": 1})  # encode: N=0, Y=1

# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# ---------------------------
# 3) Preprocessing pipelines
# ---------------------------
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ]
)

# ---------------------------
# 4) Model
# ---------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

# Full pipeline (preprocessing + ML model)
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ---------------------------
# 5) Train/Test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# 6) Train
# ---------------------------
clf.fit(X_train, y_train)

# ---------------------------
# 7) Evaluate
# ---------------------------
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n✅ Model Training Complete!")
print(f"✅ Accuracy: {acc * 100:.2f}%\n")

print("📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 8) Save Model + Column Info
# ---------------------------
joblib.dump(clf, "loan_model.pkl")
joblib.dump({"cat_cols": cat_cols, "num_cols": num_cols}, "columns.pkl")

print("\n✅ Saved: loan_model.pkl and columns.pkl")
