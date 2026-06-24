import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# ==========================
# Load Dataset
# ==========================

df = pd.read_excel("Risk_Alert_Classifier_Dataset_4600.xlsx")

# ==========================
# Date Column
# ==========================

df["last_transaction_days"] = (
    df["last_transaction_date"].max()
    - df["last_transaction_date"]
).dt.days

df.drop("last_transaction_date", axis=1, inplace=True)

# ==========================
# Encode Categorical Columns
# ==========================

le_gender = LabelEncoder()
le_region = LabelEncoder()
le_emp = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"].astype(str))
df["region"] = le_region.fit_transform(df["region"].astype(str))
df["employment_type"] = le_emp.fit_transform(df["employment_type"].astype(str))

# ==========================
# KNN Imputer
# ==========================

imputer = KNNImputer(n_neighbors=5)

df = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# ==========================
# Features and Target
# ==========================

X = df.drop("risk_status", axis=1)
y = df["risk_status"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Logistic Regression
# ==========================

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

print("\nLOGISTIC REGRESSION")

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")
print(cm)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("AUC ROC :", roc_auc_score(y_test, y_prob))

TN, FP, FN, TP = cm.ravel()

print("Type I Error :", FP)
print("Type II Error :", FN)

# ==========================
# Under Sampling
# ==========================

rus = RandomUnderSampler(random_state=42)

X_under, y_under = rus.fit_resample(X_train, y_train)

lr_under = LogisticRegression(max_iter=1000)

lr_under.fit(X_under, y_under)

pred_under = lr_under.predict(X_test)

print("\nUnder Sampling")
print("Recall :", recall_score(y_test, pred_under))
print("F1 :", f1_score(y_test, pred_under))

# ==========================
# Over Sampling
# ==========================

ros = RandomOverSampler(random_state=42)

X_over, y_over = ros.fit_resample(X_train, y_train)

lr_over = LogisticRegression(max_iter=1000)

lr_over.fit(X_over, y_over)

pred_over = lr_over.predict(X_test)

print("\nOver Sampling")
print("Recall :", recall_score(y_test, pred_over))
print("F1 :", f1_score(y_test, pred_over))

# ==========================
# SMOTE
# ==========================

smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X_train, y_train)

lr_smote = LogisticRegression(max_iter=1000)

lr_smote.fit(X_smote, y_smote)

pred_smote = lr_smote.predict(X_test)

print("\nSMOTE")
print("Recall :", recall_score(y_test, pred_smote))
print("F1 :", f1_score(y_test, pred_smote))

# ==========================
# ADASYN
# ==========================

adasyn = ADASYN(random_state=42)

X_ada, y_ada = adasyn.fit_resample(X_train, y_train)

lr_ada = LogisticRegression(max_iter=1000)

lr_ada.fit(X_ada, y_ada)

pred_ada = lr_ada.predict(X_test)

print("\nADASYN")
print("Recall :", recall_score(y_test, pred_ada))
print("F1 :", f1_score(y_test, pred_ada))

# ==========================
# Decision Tree
# ==========================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy :",
      accuracy_score(y_test, dt_pred))

# ==========================
# Random Forest
# ==========================

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Accuracy :",
      accuracy_score(y_test, rf_pred))

# ==========================
# Randomized Search CV
# ==========================

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}

random_search = RandomizedSearchCV(
    rf,
    param_dist,
    n_iter=10,
    cv=5,
    scoring="f1",
    random_state=42
)

random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_

# ==========================
# Grid Search CV
# ==========================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15, None]
}

grid = GridSearchCV(
    best_rf,
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)

final_model = grid.best_estimator_

final_pred = final_model.predict(X_test)
final_prob = final_model.predict_proba(X_test)[:, 1]

print("\nTuned Random Forest Accuracy :",
      accuracy_score(y_test, final_pred))

print("Tuned Random Forest F1 :",
      f1_score(y_test, final_pred))

print("Tuned Random Forest AUC :",
      roc_auc_score(y_test, final_prob))

# ==========================
# ROC Curve
# ==========================

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "Tuned RF": final_model
}

plt.figure(figsize=(8, 6))

for name, model in models.items():

    prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, prob)

    auc = roc_auc_score(y_test, prob)

    plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()