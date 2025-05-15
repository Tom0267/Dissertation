import os
import cv2
import torch
import random
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    TrainingArguments,
    Trainer,
    TimesformerConfig
)
import kagglehub
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter, RandomApply
from torch.utils.data import Dataset, random_split

#configuration
modelName = "facebook/timesformer-base-finetuned-k400"
device = "cuda" if torch.cuda.is_available() else "cpu"
videoFrames = 20
frameInterval = 15
maxFrames = 32

#dataset setup
datasetPath = kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset")
labelMap = {"NonViolence": 0, "Violence": 1}

#model + processor
processor = AutoImageProcessor.from_pretrained(modelName)

#load config and force num_labels=2
config = TimesformerConfig.from_pretrained(modelName)
config.num_labels = 2
#print(config)
config.hidden_dropout_prob = 0.1
config.attention_probs_dropout_prob = 0.1

#step 2: load the model but ignore the mismatched head
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
    def __init__(self, root, labels, transform, numFrames=videoFrames, interval=frameInterval):
        self.samples = []
        self.labels = labels
        self.transform = transform
        self.numFrames = numFrames
        self.interval = interval

        for labelName in ["NonViolence", "Violence"]:
            labelPath = os.path.join(root, labelName)
            for fileName in os.listdir(labelPath):
                if fileName.endswith(".mp4"):
                    self.samples.append((os.path.join(labelPath, fileName), self.labels[labelName]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames, _, _ = read_video(path, pts_unit="sec")
        total = frames.shape[0]

        #temporal jitter
        max_start = max(0, total - (self.numFrames - 1) * self.interval)
        start = random.randint(0, max_start) if max_start > 0 else 0

        selected = []
        for i in range(self.numFrames):
            frameNum = start + i * self.interval
            if frameNum < total:
                frame = frames[frameNum]
                image = Image.fromarray(frame.numpy())
                selected.append(image)
            else:
                break  #stop if exceeded video length

        #pad with black frames if needed
        while len(selected) < maxFrames:
            selected.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        #let processor handle resizing, normalising, etc.
        with torch.no_grad():
            pixelValues = processor(images=selected, return_tensors="pt")["pixel_values"]

        return {"pixel_values": pixelValues.squeeze(0), "labels": label}

#create dataset and train/val split
dataset = ViolenceDataset(datasetPath, labelMap, transform)
trainSize = int(0.7 * len(dataset))
valSize = len(dataset) - trainSize
trainDs, valDs = random_split(dataset, [trainSize, valSize])

#training args
args = TrainingArguments(
    output_dir="./timesformer_rlvs",
    per_device_train_batch_size=1,
    eval_strategy="epoch",
    num_train_epochs=2,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    weight_decay=0.01,
    max_grad_norm=1.0    
)

#trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainDs,
    eval_dataset=valDs,
)

trainer.train()