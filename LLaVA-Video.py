import os
import socket
import torch
import shutil
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    TrainingArguments,
    Trainer,
    TimesformerConfig,
    EarlyStoppingCallback
)
import kagglehub
from torchvision.io import read_video
from sklearn.metrics import accuracy_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter, RandomApply
from torch.utils.data import Dataset, random_split


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

def accuracyScore(eval_pred):
    logits, labels = eval_pred
    preds = torch.tensor(logits).argmax(dim=-1)
    return {
        "accuracy": accuracy_score(labels, preds)
    }

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
    greater_is_better=True
)

#trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainDs,
    eval_dataset=valDs,
    compute_metrics=accuracyScore,
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
testResults = trainer.evaluate(testDs)
print("=== TEST RESULTS ===")
print(testResults.metrics)