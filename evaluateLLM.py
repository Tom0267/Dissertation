from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

#paths
resultsCsv = "RLVS/results.csv"
changesTxt = "RLVS/decision_changes.txt"
mistakesTxt = "RLVS/secondary_mistakes.txt"
plotPath = "RLVS/confusion_matrix.png"

#load data
df = pd.read_csv(resultsCsv)

#true label from ID
def inferTrueLabel(videoId):
    videoId = videoId.lower()
    if videoId.startswith("nonviolence"):
        return "non-violent"
    elif videoId.startswith("violence"):
        return "violent"
    return "unknown"

df["TrueLabel"] = df["VideoID"].apply(inferTrueLabel)
df["PrimaryPred"] = df["PrimaryDecision"].str.lower().map({"yes": "violent", "no": "non-violent"})  #map primary decisions to labels
df["FinalPred"] = df["FinalDecision"].str.lower().map({"yes": "violent", "no": "non-violent"})      #map final decisions to labels

#find changes
df["SecondaryPred"] = df["SecondaryDecision"].str.lower().map({"yes": "violent", "no": "non-violent"})  
changed = df[df["PrimaryPred"] != df["SecondaryPred"]]      
changed.to_csv(changesTxt, index=False, header=False)       #save changed decisions to csv
print(f"Written {len(changed)} changed decisions to: {changesTxt}")

#evaluate impact of changes
correctChange = 0
incorrectChange = 0
wrongChangeVideos = []

for _, row in changed.iterrows():
    true = row["TrueLabel"]
    primary = row["PrimaryPred"]
    second = row["SecondaryPred"]

    if second == true and primary != true:      #corrected a wrong label
        correctChange += 1
    elif second != true and primary == true:        #introduced a new error
        incorrectChange += 1
        wrongChangeVideos.append(row)

#save incorrect secondaries
pd.DataFrame(wrongChangeVideos).to_csv(mistakesTxt, index=False, header=False)
print(f"Saved {len(wrongChangeVideos)} incorrect secondary decisions to {mistakesTxt}")

#full evaluation
df["PredLabel"] = df["PrimaryPred"]
#y_true = df["TrueLabel"]
#y_pred = df["PredLabel"]

valid_mask = df["TrueLabel"].isin(["violent", "non-violent"]) & df["PredLabel"].isin(["violent", "non-violent"])
filtered_df = df[valid_mask]

y_true = filtered_df["TrueLabel"]
y_pred = filtered_df["PredLabel"]

print("y_true unique:", set(y_true))
print("y_pred unique:", set(y_pred))

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

df["SecondaryLabel"] = df["SecondaryPred"]
valid_secondary_mask = df["TrueLabel"].isin(["violent", "non-violent"]) & df["SecondaryLabel"].isin(["violent", "non-violent"])
secondYTrue = df[valid_secondary_mask]["TrueLabel"]
secondYPred = df[valid_secondary_mask]["SecondaryLabel"]

secondAcc = accuracy_score(secondYTrue, secondYPred)
secondPrecision = precision_score(secondYTrue, secondYPred, pos_label="violent")
secondRecall = recall_score(secondYTrue, secondYPred, pos_label="violent")
secondF1 = f1_score(secondYTrue, secondYPred, pos_label="violent")

print("\n=== Decision Change Impact ===")
print(f"Total changed decisions   : {len(changed)}")
print(f" - Corrected wrong labels : {correctChange}")
print(f" - Introduced new errors  : {incorrectChange}\n")
print(f"Secondary decision accuracy: {secondAcc:.4f}")
print(f"Secondary decision precision: {secondPrecision:.4f}")
print(f"Secondary decision recall: {secondRecall:.4f}")
print(f"Secondary decision F1 score: {secondF1:.4f}")

#plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["violent", "non-violent"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(plotPath)
plt.close()
print(f"Confusion matrix saved to {plotPath}")