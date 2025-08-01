from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

#paths
resultsCsv = "RWF-2000/DeepseekResults.csv"
plotPath = "RWF-2000/Deepseek_confusion_matrix.png"

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

#plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["violent", "non-violent"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(plotPath)
plt.close()
print(f"Confusion matrix saved to {plotPath}")