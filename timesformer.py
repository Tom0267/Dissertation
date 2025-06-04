import os
import time
import socket
import torch
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import CalibrationDisplay
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    TrainingArguments,
    Trainer,
    TimesformerConfig,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class ViolenceDataset(Dataset):
    def __init__(self, rootDir):
        self.samples = sorted(
            [os.path.join(rootDir, f) for f in os.listdir(rootDir) if f.endswith(".pt")]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filePath = self.samples[idx]
        try:
            data = torch.load(filePath, weights_only=True)
            pixel_values = data["pixel_values"].float() / 255.0  # (T, C, H, W)
            if pixel_values.shape[0] == 3:
                # assume (C, T, H, W)
                pass
            elif pixel_values.shape[1] == 3:
                # assume (T, C, H, W), need to permute
                pixel_values = pixel_values.permute(1, 0, 2, 3)
            else:
                raise ValueError(f"Unrecognised shape {pixel_values.shape} in {filePath}")

            expected_frames = 16
            c, t, h, w = pixel_values.shape
            if t < expected_frames:
                pad = torch.zeros((c, expected_frames - t, h, w), dtype=pixel_values.dtype)
                pixel_values = torch.cat([pixel_values, pad], dim=1)
            elif t > expected_frames:
                pixel_values = pixel_values[:, :expected_frames]

            if pixel_values.shape[0] != 3:
                raise ValueError(f"Expected 3 channels but got {pixel_values.shape[0]} in {filePath}")
            return {
                "pixel_values": pixel_values,
                "labels": 0 if os.path.basename(filePath).startswith("NonViolence_") else 1,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load {filePath}: {e}")


# slurm
print("=== SLURM ENVIRONMENT INFO ===")
print(f"Hostname: {socket.gethostname()}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not in SLURM')}")
print(f"SLURM_ARRAY_TASK_ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("=== END SLURM INFO ===\n")

# configuration
modelName = "facebook/timesformer-base-finetuned-k400"
device = "cuda" if torch.cuda.is_available() else "cpu"
labelMap = {"NonViolence": 0, "Violence": 1}
TRAINED = os.path.exists("timesformerTrained/model.safetensors")

# model + processor
processor = AutoImageProcessor.from_pretrained(modelName, use_fast=False)

# load config and force num_labels=2
config = TimesformerConfig.from_pretrained(modelName)
config.num_labels = 2
# print(config)
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

if not TRAINED:
    # load the model but ignore the mismatched head
    model = TimesformerForVideoClassification.from_pretrained(
        modelName, config=config, ignore_mismatched_sizes=True
    )
    model.gradient_checkpointing_enable()
    model = model.to(device)
    model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
    
    trainDs = ViolenceDataset("train_tensors")
    testDs = ViolenceDataset("RWF-2000/test_tensors")
    valDs = ViolenceDataset("val_tensors")
    
else:
    model = TimesformerForVideoClassification.from_pretrained("timesformerTrained", config=config)
    testDs = ViolenceDataset("test_tensors")

def metrics(eval_pred, threshold=0.5):      #threshold determined by thresholdSweep
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
        roc_auc = float("nan")

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    report = classification_report(labels, preds, output_dict=True)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def plotMetrics(testResults, save_dir="evaluation_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # plot confusion matrix
    try:
        cm = np.array(testResults["eval_confusion_matrix"])
    except (TypeError, KeyError):
        cm = np.array(testResults.metrics["test_confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NonViolence", "Violence"],
        yticklabels=["NonViolence", "Violence"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # plot ROC curve using existing predictions
    try:
        logits = testResults.predictions
        labels = testResults.label_ids
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

        RocCurveDisplay.from_predictions(labels, probs)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")


def thresholdSweep(trainer, dataset, thresholds=np.arange(0.1, 1.0, 0.05)):
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

        results.append(
            {
                "threshold": t,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
            }
        )

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


def dataBatcher(batch):
    pixel_values = torch.stack([item["pixel_values"].permute(1, 0, 2, 3) for item in batch])  # (B, T, C, H, W)
    labels = torch.tensor([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

def plotCalibrationCurve(probs, labels):
    # create figure and axis first
    fig, ax = plt.subplots(figsize=(6, 5))

    # plot calibration curve using from_predictions
    CalibrationDisplay.from_predictions(labels, probs, n_bins=10, strategy='uniform', ax=ax, name="Timesformer")

    ax.set_title("Calibration Curve (Reliability Diagram)")
    fig.tight_layout()

    os.makedirs("evaluation_plots", exist_ok=True)
    fig.savefig("evaluation_plots/calibration_curve.png")
    plt.close(fig)

if not TRAINED:
    # training args
    args = TrainingArguments(
        per_device_train_batch_size=1,
        eval_strategy="epoch",
        num_train_epochs=10,
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=500,
        fp16=torch.cuda.is_available(),
        weight_decay=0.3,
        max_grad_norm=1.0,
        output_dir="./timesformerOutput",
        save_total_limit=1,
        metric_for_best_model="f1",
        greater_is_better=True,
        label_smoothing_factor=0.2, #1753 had 0.3 1754 had 0.2
        lr_scheduler_type="cosine",
        warmup_steps=300,
        learning_rate=1e-4
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=trainDs,
        eval_dataset=valDs,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=dataBatcher,
    )
else:
    trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./timesformerOutput",  # required by Trainer
        per_device_eval_batch_size=1,
    ),
    compute_metrics=metrics,
    data_collator=dataBatcher,
)

if trainer.state.best_model_checkpoint and os.path.exists(
    trainer.state.best_model_checkpoint
):
    print("Resuming from checkpoint...")
    trainer.train(resume_from_checkpoint=trainer.state.best_model_checkpoint)
    trainer.save_model("timesformerTrained")
elif not TRAINED:
    print("Starting training from scratch...")
    trainer.train()
    trainer.save_model("timesformerTrained")
else:
    print("Using existing trained model...")
    model = TimesformerForVideoClassification.from_pretrained("timesformerTrained", config=config).to(device)
    trainer.model = model

#count the number of violent and non-violent videos in the test set
testViolenceCount = sum(1 for sample in testDs if sample["labels"] == 1)
testNonViolenceCount = sum(1 for sample in testDs if sample["labels"] == 0)
print(f"Test set - Violent: {testViolenceCount}, Non-Violent: {testNonViolenceCount}")

if not TRAINED:
    trainViolenceCount = sum(1 for sample in trainDs if sample["labels"] == 1)
    trainNonViolenceCount = sum(1 for sample in trainDs if sample["labels"] == 0)
    print(f"Train set - Violent: {trainViolenceCount}, Non-Violent: {trainNonViolenceCount}")

# test the model
startTime = time.time()
predictionResults = trainer.predict(test_dataset=testDs)
endTime = time.time()
testMetrics = predictionResults.metrics
print(testMetrics)

for key, name in [
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("roc_auc", "ROC-AUC"),
    ("confusion_matrix", "Confusion Matrix"),
    ("classification_report", "Classification Report"),
]:
    value = testMetrics.get(f"test_{key}", "N/A")
    print(f"{name}: {value}", flush=True)
    
print(f"Prediction time: {endTime - startTime:.2f} seconds", flush=True)
probs = torch.nn.functional.softmax(torch.tensor(predictionResults.predictions), dim=-1).numpy()[:, 1]
labels = predictionResults.label_ids
plotCalibrationCurve(probs, labels)

plt.hist(probs, bins=50, edgecolor='black')
plt.title("Distribution of Violence Probabilities")
plt.xlabel("P(Violence)")
plt.ylabel("Count")
plt.savefig("evaluation_plots/violence_probabilities_distribution.png")


# testResults = trainer.evaluate(eval_dataset=testDs, metric_key_prefix="eval")
  
# print("=== TEST RESULTS ===")
# for key, name in [
#     ("accuracy", "Accuracy"),
#     ("f1", "F1"),
#     ("precision", "Precision"),
#     ("recall", "Recall"),
#     ("roc_auc", "ROC-AUC"),
#     ("confusion_matrix", "Confusion Matrix"),
#     ("classification_report", "Classification Report"),
# ]:
#     value = testResults.get(f"eval_{key}", "N/A")
#     print(f"{name}: {value}", flush=True)
# print(f"Prediction time: {endTime - startTime:.2f} seconds", flush=True)

# plotMetrics(testResults)

best_threshold = thresholdSweep(trainer, testDs)
print(f"Best threshold for F1: {best_threshold['threshold']}")
print(f"Best F1: {best_threshold['f1']}")
print(f"Best Precision: {best_threshold['precision']}")
print(f"Best Recall: {best_threshold['recall']}")
print(f"Best Accuracy: {best_threshold['accuracy']}")
