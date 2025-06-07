from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
import os

#config
trainFeaturePath = "RLVS/motion_features.json"
trainSplitPath = "RLVS/split.json"
testFeaturePath = "RLVS/motion_features.json"
featurePath = "RLVS/motion_features.json"
#testFeaturePath = "RWF-2000/motion_features.json"
#featurePath = "RWF-2000/motion_features.json"
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

#load RWF-2000 test features
with open(testFeaturePath, "r") as f:
    featuresByPathTest = json.load(f)

def getLabel(key: str) -> int | None:       #map video IDs to labels
    name = os.path.basename(key).lower() if "/" in key or "\\" in key else key.lower()  #handle both path separators
    prefix = name.split("_")[0]                         #get the prefix before the first underscore
    if prefix in {"nonviolence", "nofights", "nv"}:
        return 0
    elif prefix in {"violence", "fights", "v"}:
        return 1
    return None

def extractFeatures(featuresMap, splitMap=None, splitType=None):
    X, y = [], []
    for key, features in featuresMap.items():                           #iterate over all features
        if splitMap and splitType and splitMap.get(key) != splitType:   #filter by split type if provided
            continue
        label = getLabel(key)   #get label for the video ID
        if label is None:
            print(f"Skipping unlabelled {splitType or 'unknown'} key: {key}")   #skip if label is None
            continue
        X.append(list(features.values()))   #append feature values to X
        y.append(label)                  #append label to y
    return X, y

#get feature names from training set
featureNames = list(next(iter(featuresByPathTrain.values())).keys())

#build training and test sets
XTrain, yTrain = extractFeatures(featuresByPathTrain)
XTest, yTest = extractFeatures(featuresByPathTest)
    
#sanity check
print(testFeaturePath)
if ((not XTest or not XTrain) and testFeaturePath.startswith("RLVS")):
    raise RuntimeError("One of the splits is empty. Check your preprocessing step.")

#grid search parameters
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [None, 5, 8, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'class_weight': ['balanced', None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=2,
    n_jobs=-1
)

#grid.fit(XTrain, yTrain)
#best parameters
#print("Best parameters found: ", grid.best_params_)
#print("Best ROC AUC score: ", grid.best_score_)

#train using best parameters
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",        #use balanced class weights to handle class imbalance
    random_state=42,
    n_jobs=-1
)

#fit the model and make predictions
clf.fit(XTrain, yTrain)
startTime = time.time()
y_pred = clf.predict(XTest)
y_proba = clf.predict_proba(XTest)[:, 1]
endTime = time.time()

#evaluation
print("\n=== Random Forest Evaluation ===")
print("Confusion Matrix:\n", confusion_matrix(yTest, y_pred))
print("\nClassification Report:\n", classification_report(yTest, y_pred))
print("Accuracy:", accuracy_score(yTest, y_pred))
print("F1 Score:", f1_score(yTest, y_pred))
print("Precision:", precision_score(yTest, y_pred))
print("Recall:", recall_score(yTest, y_pred))
print("ROC AUC:", roc_auc_score(yTest, y_proba))
print(f"Prediction time: {endTime - startTime:.2f} seconds")

if not os.path.exists("random_forest"):
    os.makedirs("random_forest")

#plot feature importance
importances = clf.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(featureNames, importances)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("random_forest/random_forest_feature_importance.png")

#plot ROC curve
RocCurveDisplay.from_predictions(yTest, y_proba)
plt.title("ROC Curve - Random Forest")
plt.tight_layout()
plt.savefig("random_forest/random_forest_roc_curve.png")

trainDf = pd.DataFrame(XTrain, columns=featureNames)
testDf = pd.DataFrame(XTest, columns=featureNames)

for feature in featureNames:
    sns.kdeplot(trainDf[feature], label='Train', fill=True)
    sns.kdeplot(testDf[feature], label='Test', fill=True)
    plt.title(f"Feature: {feature}")
    plt.legend()
    plt.show()