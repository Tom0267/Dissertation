from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import kagglehub
import json
import time
import os

#config
trainFeaturePath = "RLVS/motion_features.json"
trainSplitPath = "RLVS/split.json"
testFeaturePath = "RLVS/motion_features.json"
featurePath = "RLVS/motion_features.json"
#splitPath = "split.json"
labelMap = {"Violence": 1, "NonViolence": 0}

#load features and split info
with open(featurePath, "r") as f:
    featuresByPath = json.load(f)

with open(trainSplitPath, "r") as f:
    splitMap = json.load(f)

#prepare data
#load RLVS training features
with open(trainFeaturePath, "r") as f:
    featuresByPathTrain = json.load(f)

#load Hockey Fights test features
with open(testFeaturePath, "r") as f:
    featuresByPathTest = json.load(f)

#get feature names from training set
featureNames = list(next(iter(featuresByPathTrain.values())).keys())
X_train, y_train = [], []
for fullPath, features in featuresByPathTrain.items():
    if fullPath not in splitMap or splitMap[fullPath] != "train":
        continue

    labelFolder = os.path.basename(os.path.dirname(fullPath)).lower()
    if labelFolder in ["violence", "fights"]:
        label = 1
    elif labelFolder in ["nonviolence", "nofights"]:
        label = 0
    else:
        print(f"Skipping unlabelled training key: {fullPath}")
        continue

    X_train.append(list(features.values()))
    y_train.append(label)
    
print(X_train[:5])
print(y_train[:5])

X_test, y_test = [], []
for name, features in featuresByPathTest.items():

    filename = os.path.basename(name)  # "NV_993.mp4"
    labelPrefix = filename.split("_")[0].lower()
    if name not in splitMap or splitMap[name] != "test":
        continue
    if labelPrefix in ["violence", "fights", "v"]:
        label = 1
    elif labelPrefix in ["nonviolence", "nofights", "nv"]:
        label = 0
    else:
        print(f"Skipping unlabelled test key: {name}")
        continue

    X_test.append(list(features.values()))
    y_test.append(label)
    
print(X_test[:5])  #print first 5 test samples
print(y_test[:5])  #print first 5 test labels

#for path, features in featuresByPath.items():
#    if path not in splitMap:
#        continue  #skip if not in split
#    labelStr = os.path.basename(os.path.dirname(path))
#    label = labelMap[labelStr]
#    featVec = list(features.values())

#    if splitMap[path] == "train":
#        X_train.append(featVec)
#        y_train.append(label)
#    elif splitMap[path] == "test":
#        X_test.append(featVec)
#        y_test.append(label)

#sanity check
if not X_test or not X_train:
    raise RuntimeError("One of the splits is empty. Check your preprocessing step.")

#train using best parameters
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

#evaluation
print("\n=== Random Forest Evaluation ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(f"Prediction time: {endTime - startTime:.2f} seconds")

#plot feature importance
importances = clf.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(featureNames, importances)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

#plot ROC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve - Random Forest")
plt.tight_layout()
plt.savefig("random_forest_roc_curve.png")

if not os.path.exists("feature_importances"):
    os.makedirs("feature_importances")
plt.figure(figsize=(8, 5))
plt.barh(featureNames, importances)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png")
