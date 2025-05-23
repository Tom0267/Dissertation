from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

#paths
resultsCsv = "results.csv"
changesTxt = "decision_changes.txt"
mistakesTxt = "secondary_mistakes.txt"
plotPath = "confusion_matrix.png"

#load data
df = pd.read_csv(resultsCsv)

#true label from ID
def inferTrueLabel(videoId):
    if videoId.startswith("NV_"):
        return "non-violent"
    elif videoId.startswith("V_"):
        return "violent"
    return "unknown"

df["TrueLabel"] = df["VideoID"].apply(inferTrueLabel)
df["PrimaryPred"] = df["PrimaryDecision"].str.lower().map({"yes": "violent", "no": "non-violent"})
df["FinalPred"] = df["FinalDecision"].str.lower().map({"yes": "violent", "no": "non-violent"})

#find changes
changed = df[df["PrimaryPred"] != df["FinalPred"]]
changed.to_csv(changesTxt, index=False, header=False)
print(f"Written {len(changed)} changed decisions to: {changesTxt}")

#evaluate impact of changes
correctChange = 0
incorrectChange = 0
wrongChangeVideos = []

for _, row in changed.iterrows():
    true = row["TrueLabel"]
    primary = row["PrimaryPred"]
    final = row["FinalPred"]

    if final == true and primary != true:
        correctChange += 1
    elif final != true and primary == true:
        incorrectChange += 1
        wrongChangeVideos.append(row)

#save incorrect secondaries
pd.DataFrame(wrongChangeVideos).to_csv(mistakesTxt, index=False, header=False)
print(f"Saved {len(wrongChangeVideos)} incorrect secondary decisions to {mistakesTxt}")

#full evaluation
df["PredLabel"] = df["FinalPred"]
y_true = df["TrueLabel"]
y_pred = df["PredLabel"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="violent")
recall = recall_score(y_true, y_pred, pos_label="violent")
f1 = f1_score(y_true, y_pred, pos_label="violent")
cm = confusion_matrix(y_true, y_pred, labels=["violent", "non-violent"])

#output
print("\n=== Evaluation Summary ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix:\n", cm)

print("\n=== Decision Change Impact ===")
print(f"Total changed decisions   : {len(changed)}")
print(f" - Corrected wrong labels : {correctChange}")
print(f" - Introduced new errors  : {incorrectChange}")

#plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["violent", "non-violent"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(plotPath)
plt.close()
print(f"Confusion matrix saved to {plotPath}")