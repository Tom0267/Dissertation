import os
import socket
import torch
import shutil
import kagglehub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from torch.utils.data import Dataset, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter, RandomApply
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    TrainingArguments,
    Trainer,
    TimesformerConfig,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

#slurm
print("=== SLURM ENVIRONMENT INFO ===")
print(f"Hostname: {socket.gethostname()}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not in SLURM')}")
print(f"SLURM_ARRAY_TASK_ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("=== END SLURM INFO ===\n")
    
#configuration
modelName = "facebook/timesformer-base-finetuned-k400"
device = "cuda" if torch.cuda.is_available() else "cpu"
videoFrames = 20
frameInterval = 15
maxFrames = 32

#dataset setup
datasetPath = kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset")
datasetPath = os.path.join(datasetPath, "Real Life Violence Dataset")
labelMap = {"NonViolence": 0, "Violence": 1}

#model + processor
processor = AutoImageProcessor.from_pretrained(modelName, use_fast=False)

#load config and force num_labels=2
config = TimesformerConfig.from_pretrained(modelName)
config.num_labels = 2
#print(config)
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

#load the model but ignore the mismatched head
model = TimesformerForVideoClassification.from_pretrained(
    modelName,
    config=config,
    ignore_mismatched_sizes=True
)
model.gradient_checkpointing_enable()
model = model.to(device)

#preprocessing for each frame
transform = Compose([
    Resize((224, 224)),
    RandomApply([ColorJitter(brightness=0.2)], p=0.2),
    RandomApply([ColorJitter(contrast=0.2)], p=0.2),
    RandomApply([ColorJitter(saturation=0.2)], p=0.2),
    RandomApply([ColorJitter(hue=0.1)], p=0.2),
    ToTensor()
])

class ViolenceDataset(Dataset):
    def __init__(self, rootDir):
        self.samples = sorted([
            os.path.join(rootDir, f)
            for f in os.listdir(rootDir)
            if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filePath = self.samples[idx]
        try:
            data = torch.load(filePath)
            return {
                "pixel_values": data["pixel_values"],
                "labels": data["label"]
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load {filePath}: {e}")

#create dataset and train/val split
dataset = ViolenceDataset("preprocessed")
fullSize = len(dataset)
trainSize = int(0.7 * fullSize)
valSize = int(0.15 * fullSize)
testSize = fullSize - trainSize - valSize

trainDs, valDs, testDs = random_split(dataset, [trainSize, valSize, testSize])

# copy files to test_videos
testFilePaths = [dataset.samples[i] for i in testDs.indices]
os.makedirs("test_videos", exist_ok=True)

for src in testFilePaths:
    dst = os.path.join("test_videos", os.path.basename(src))
    shutil.copyfile(src, dst)

def metrics(eval_pred, threshold=0.5):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = (probs[:, 1] >= threshold).astype(int)
    labels = np.array(labels)

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)

    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        roc_auc = float('nan')

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
def plotMetrics(results, save_dir="evaluation_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # confusion matrix
    cm = np.array(results["eval_confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["NonViolence", "Violence"], yticklabels=["NonViolence", "Violence"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # ROC curve
    try:
        # re-evaluate to get raw logits and labels
        outputs = trainer.predict(testDs)
        logits = outputs.predictions
        labels = outputs.label_ids
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        RocCurveDisplay.from_predictions(labels, probs)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")

def threshold_sweep(trainer, dataset, thresholds=np.arange(0.1, 1.0, 0.05)):
    outputs = trainer.predict(dataset)
    logits = outputs.predictions
    labels = outputs.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        accuracy = accuracy_score(labels, preds)

        results.append({
            "threshold": t,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        })

    # plot
    plt.figure(figsize=(8, 5))
    for metric in ["f1", "precision", "recall", "accuracy"]:
        plt.plot(thresholds, [r[metric] for r in results], label=metric)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("evaluation_plots/threshold_tuning.png")
    plt.close()

    # return best threshold based on highest F1
    best = max(results, key=lambda x: x["f1"])
    print(f"Best Threshold (F1): {best['threshold']} with F1 = {best['f1']:.4f}")
    return best


#training args
args = TrainingArguments(
    per_device_train_batch_size=1,
    eval_strategy="epoch",
    num_train_epochs=10,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    weight_decay=0.05,
    max_grad_norm=1.0,
    output_dir="./timesformerOutput",
    save_total_limit=2,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    
)

#trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainDs,
    eval_dataset=valDs,
    compute_metrics=metrics,
    callbacks= [EarlyStoppingCallback(early_stopping_patience=3)]
)

if trainer.state.best_model_checkpoint and os.path.exists(trainer.state.best_model_checkpoint):
    print("Resuming from checkpoint...")
    trainer.train(resume_from_checkpoint=trainer.state.best_model_checkpoint)
else:
    print("Starting training from scratch...")
    trainer.train()

trainer.save_model("timesformerTrained")

trainer.evaluate()

#test the model
testResults = trainer.evaluate(eval_dataset=testDs)
print("=== TEST RESULTS ===")
for key, name in [
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("roc_auc", "ROC-AUC"),
    ("confusion_matrix", "Confusion Matrix"),
    ("classification_report", "Classification Report")
]:
    print(f"{name}: {testResults.get(f'eval_{key}', 'N/A')}")
    
plotMetrics(testResults, save_dir="evaluation_plots")

best_threshold = threshold_sweep(trainer, testDs)
print(f"Best threshold for F1: {best_threshold['threshold']}")
print(f"Best F1: {best_threshold['f1']}")
print(f"Best Precision: {best_threshold['precision']}")
print(f"Best Recall: {best_threshold['recall']}")
print(f"Best Accuracy: {best_threshold['accuracy']}")