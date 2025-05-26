from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import kagglehub
import json
import time
import os

# config
featurePath = "motion_features.json"
splitPath = "split.json"
labelMap = {"Violence": 1, "NonViolence": 0}

# load features and split info
with open(featurePath, "r") as f:
    featuresByPath = json.load(f)

with open(splitPath, "r") as f:
    splitMap = json.load(f)

# prepare data
X_train, y_train, X_test, y_test = [], [], [], []
featureNames = list(next(iter(featuresByPath.values())).keys())

for path, features in featuresByPath.items():
    if path not in splitMap:
        continue  # skip if not in split
    labelStr = os.path.basename(os.path.dirname(path))
    label = labelMap[labelStr]
    featVec = list(features.values())

    if splitMap[path] == "train":
        X_train.append(featVec)
        y_train.append(label)
    elif splitMap[path] == "test":
        X_test.append(featVec)
        y_test.append(label)

# sanity check
if not X_test or not X_train:
    raise RuntimeError("One of the splits is empty. Check your preprocessing step.")

# train using best parameters
clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)
startTime = time.time()
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
endTime = time.time()

# evaluation
print("\n=== Random Forest Evaluation ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(f"Prediction time: {endTime - startTime:.2f} seconds")

# plot feature importance
importances = clf.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(featureNames, importances)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# plot ROC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve - Random Forest")
plt.tight_layout()
plt.show()