import os
import time
import socket
import torch
import numpy as np
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, log_loss
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
    classification_report,
)

class ViolenceDataset(Dataset):
    def __init__(self, rootDir, valid_ids_file=None):  # now optional
        if valid_ids_file and os.path.exists(valid_ids_file):
            with open(valid_ids_file, "r") as f:
                allowed_ids = set(line.strip() for line in f if line.strip())
        else:
            allowed_ids = None  # no filtering

        all_samples = [
            os.path.join(rootDir, f) for f in os.listdir(rootDir) if f.endswith(".pt")
        ]

        if allowed_ids:
            self.samples = sorted([
                f for f in all_samples if os.path.splitext(os.path.basename(f))[0].lower() in allowed_ids
            ])
        else:
            self.samples = sorted(all_samples)

        print(f"[{rootDir}] Loaded {len(self.samples)} samples"
              f"{' (filtered)' if allowed_ids else ''}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filePath = self.samples[idx]
        try:
            data = torch.load(filePath, weights_only=True)
            pixelValues = data["pixel_values"].float() / 255.0

            if pixelValues.shape[0] == 3:
                pass
            elif pixelValues.shape[1] == 3:
                pixelValues = pixelValues.permute(1, 0, 2, 3)
            else:
                raise ValueError(f"Unrecognised shape {pixelValues.shape} in {filePath}")

            expected_frames = 16
            c, t, h, w = pixelValues.shape
            if t < expected_frames:
                pad = torch.zeros((c, expected_frames - t, h, w), dtype=pixelValues.dtype)
                pixelValues = torch.cat([pixelValues, pad], dim=1)
            elif t > expected_frames:
                pixelValues = pixelValues[:, :expected_frames]

            if pixelValues.shape[0] != 3:
                raise ValueError(f"Expected 3 channels but got {pixelValues.shape[0]} in {filePath}")

            label = 0 if os.path.basename(filePath).startswith("NonViolence_") else 1
            return {"pixel_values": pixelValues, "labels": label}
        except Exception as e:
            raise RuntimeError(f"Failed to load {filePath}: {e}")

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
labelMap = {"NonViolence": 0, "Violence": 1}
TRAINED = os.path.exists("timesformerTrained/model.safetensors")        #check if trained model exists
THRESHOLD = 0.2

#model + processor
processor = AutoImageProcessor.from_pretrained(modelName, use_fast=False)

#load config and force num_labels=2
config = TimesformerConfig.from_pretrained(modelName)
config.num_labels = 2         
#print(config)
config.hidden_dropout_prob = 0.3
config.attention_probs_dropout_prob = 0.3

if not TRAINED:
    #load the model but ignore the mismatched head
    model = TimesformerForVideoClassification.from_pretrained(
        modelName, config=config, ignore_mismatched_sizes=True
    )
    model.gradient_checkpointing_enable()   #enable gradient checkpointing for memory efficiency
    model = model.to(device)
    
    model.classifier = torch.nn.Linear(model.config.hidden_size, 2)     #set classifier to 2 classes (violence, non-violence)
    
    trainDs = ViolenceDataset("RLVS/train_tensors")
    #testDs = ViolenceDataset("RLVS/test_tensors")
    testDs = ViolenceDataset("RWF-2000/test_tensors", valid_ids_file="tested_video_ids.txt")  #filtered to match test set from LLM
    valDs = ViolenceDataset("RLVS/val_tensors")
    
else:
    model = TimesformerForVideoClassification.from_pretrained("timesformerTrained", config=config)
    #testDs = ViolenceDataset("RWF-2000/test_tensors", valid_ids_file="tested_video_ids.txt")  #filtered to match test set from LLM
    testDs = ViolenceDataset("RLVS/test_tensors")

def metrics(eval_pred, threshold=THRESHOLD):      #threshold determined by thresholdSweep
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()   #convert logits to probabilities
    #preds = (probs[:, 1] >= threshold).astype(int)  #convert probabilities to binary predictions based on threshold
    preds = probs.argmax(axis=1)
    labels = np.array(labels)

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)

    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
        ll = log_loss(labels, probs[:, 1])
    except ValueError:
        roc_auc = float("nan")
        ll = float("nan")

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
        "log_loss": ll
    }

def weightedLoss(outputs, labels, num_items_in_batch=None):
    # class weights: [non-violence, violence]
    weights = torch.tensor([2.5, 1], dtype=torch.float32).to(outputs.logits.device)
    return torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)

def gridSearchClassWeights(trainerTemplate, val_dataset, weight_grid):
    results = []

    for w0, w1 in weight_grid:
        print(f"\nTesting weights: NonViolence={w0}, Violence={w1}")
        
        trainer = deepcopy(trainerTemplate)
        trainer.compute_loss = weightedLoss([w0, w1])  # <- uses custom weighted loss
        
        eval_results = trainer.evaluate(eval_dataset=val_dataset)
        eval_results['weights'] = (w0, w1)
        results.append(eval_results)
    
    # sort by best F1
    results.sort(key=lambda r: r['eval_f1'], reverse=True)
    print("\nBest weight config:", results[0]['weights'])
    return results

def plotMetrics(testResults, save_dir="evaluation_plots"):  
    os.makedirs(save_dir, exist_ok=True)

    #plot ROC curve using existing predictions
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

def dataBatcher(batch):
    pixel_values = torch.stack([item["pixel_values"].permute(1, 0, 2, 3) for item in batch])  #(B, T, C, H, W)
    labels = torch.tensor([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

def plotCalibrationCurve(probs, labels):
    #create figure and axis first
    fig, ax = plt.subplots(figsize=(6, 5))

    #plot calibration curve using from_predictions
    CalibrationDisplay.from_predictions(labels, probs, n_bins=10, strategy='uniform', ax=ax, name="Timesformer")

    ax.set_title("Calibration Curve (Reliability Diagram)")
    fig.tight_layout()

    fig.savefig("evaluation_plots/calibration_curve.png")
    plt.close(fig)

if not TRAINED:
    arguments = TrainingArguments(
        per_device_train_batch_size=6,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        num_train_epochs=10,
        logging_dir="./logs",
        save_strategy="steps",
        load_best_model_at_end=True,
        logging_steps=1000,
        fp16=torch.cuda.is_available(),
        weight_decay=0.2,
        max_grad_norm=1.0,
        output_dir="./timesformerOutput",
        save_total_limit=1,
        metric_for_best_model="log_loss",
        greater_is_better=False,
        label_smoothing_factor=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.25,
        learning_rate=5e-5,
        seed=42
    )

    #trainer
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=trainDs,
        eval_dataset=valDs,
        compute_metrics=metrics,
        compute_loss_func=weightedLoss,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        data_collator=dataBatcher,
    )
else:
    trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./timesformerOutput",  #required by Trainer
        per_device_eval_batch_size=1,
    ),
    compute_metrics=metrics,
    data_collator=dataBatcher,
)

if trainer.state.best_model_checkpoint and os.path.exists(trainer.state.best_model_checkpoint ): #check model checkpoint exists
    print("Resuming from checkpoint...")
    trainer.train(resume_from_checkpoint=trainer.state.best_model_checkpoint)
    trainer.save_model("timesformerTrained")
elif not TRAINED:                                       #if not trained, train from scratch
    print("Starting training from scratch...")
    trainer.train()
    trainer.save_model("timesformerTrained")
else:                                                   #if trained model exists, load it        
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

#test the model
os.makedirs("evaluation_plots", exist_ok=True)
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
    ("log_loss", "Log Loss"),
]:
    value = testMetrics.get(f"test_{key}", "N/A")
    print(f"{name}: {value}", flush=True)
    
#save CM
cm = np.array(testMetrics["test_confusion_matrix"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NonViolence", "Violence"],
            yticklabels=["NonViolence", "Violence"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("evaluation_plots/confusion_matrix.png")
plt.close()
    
print(f"Prediction time: {endTime - startTime:.2f} seconds", flush=True)
probs = torch.nn.functional.softmax(torch.tensor(predictionResults.predictions), dim=-1).numpy()[:, 1]
labels = predictionResults.label_ids
plotCalibrationCurve(probs, labels)

plt.hist(probs, bins=50, edgecolor='black')
plt.title("Distribution of Violence Probabilities")
plt.xlabel("P(Violence)")
plt.ylabel("Count")
plt.savefig("evaluation_plots/violence_probabilities_distribution.png")

#distribution of non-violent
probs = torch.nn.functional.softmax(torch.tensor(predictionResults.predictions), dim=-1).numpy()[:, 0]  #probabilities for non-violence
plt.figure(figsize=(8, 5))
plt.hist(1 - probs, bins=50, edgecolor='black')
plt.title("Distribution of Non-Violence Probabilities")
plt.xlabel("P(Non-Violence)")
plt.ylabel("Count")
plt.savefig("evaluation_plots/non_violence_probabilities_distribution.png")    
    
# def weightedLoss(weights):
#     def compute_loss(model, inputs, return_outputs=False):
#         labels = inputs["labels"]
#         outputs = model(**inputs)
#         weight_tensor = torch.tensor(weights, dtype=torch.float32).to(outputs.logits.device)
#         loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight_tensor)
#         return (loss, outputs) if return_outputs else loss
#     return compute_loss
    
# def createTrainerTemplate(model, args, trainDs, valDs, metrics_fn, data_collator):
#     return Trainer(
#         model=model,
#         args=args,
#         train_dataset=trainDs,
#         eval_dataset=valDs,
#         compute_metrics=metrics_fn,
#         #compute_loss_func=weightedLoss,
#         data_collator=data_collator,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
#     )

# nonviolenceWeights = np.arange(1, 3.0, 0.05)
# violenceWeights = np.arange(1, 3.0, 0.05)
# weightGrid = [(nv, v) for nv in nonviolenceWeights for v in violenceWeights]

# trainerTemplate = createTrainerTemplate(
#     model=model,
#     args=arguments,
#     trainDs=trainDs,
#     valDs=valDs,
#     metrics_fn=metrics,
#     data_collator=dataBatcher
# )

# results = gridSearchClassWeights(trainerTemplate, val_dataset=valDs, weight_grid=weightGrid)

# for res in results:
#     print(f"Weights: {res['weights']}, F1: {res['eval_f1']:.4f}, Accuracy: {res['eval_accuracy']:.4f}")