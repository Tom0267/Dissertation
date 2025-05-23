import os
import torch
import random
import socket
from tqdm import tqdm
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, ToTensor, ColorJitter, RandomApply
from transformers import AutoImageProcessor
import kagglehub

# slurm info
print("=== SLURM ENVIRONMENT INFO ===")
print(f"Hostname: {socket.gethostname()}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not in SLURM')}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("=== END SLURM INFO ===\n")

# configuration
modelName = "facebook/timesformer-base-finetuned-k400"
device = "cuda" if torch.cuda.is_available() else "cpu"
videoFrames = 20
frameInterval = 15
maxFrames = 32
outputDir = "preprocessed"

os.makedirs(outputDir, exist_ok=True)

# processor
processor = AutoImageProcessor.from_pretrained(modelName, use_fast=False)

# transform
transform = Compose([
    Resize((224, 224)),
    RandomApply([ColorJitter(brightness=0.2)], p=0.2),
    RandomApply([ColorJitter(contrast=0.2)], p=0.2),
    RandomApply([ColorJitter(saturation=0.2)], p=0.2),
    RandomApply([ColorJitter(hue=0.1)], p=0.2),
    ToTensor()
])

# initial file check
print("\n=== File check ===")
allPtFiles = [f for f in os.listdir(outputDir) if f.endswith(".pt")]
print(f"Found {len(allPtFiles)} .pt files in '{outputDir}'")

# dataset
datasetPath = kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset")
datasetPath = os.path.join(datasetPath, "Real Life Violence Dataset")
labelMap = {"NonViolence": 0, "Violence": 1}

# process .mp4s to tensors
def processAndSave(className, label):
    classPath = os.path.join(datasetPath, className)
    videos = [v for v in os.listdir(classPath) if v.endswith(".mp4")]

    print(f"\n=== Checking for previously processed {className} files ===")

    # match .pt files with expected .mp4 stems
    existing = {
        f[len(className) + 1:].replace(".mp4", "").replace(".pt", "").lower()
        for f in os.listdir(outputDir)
        if f.startswith(f"{className}_") and f.endswith(".pt")
    }

    toSkip = [v for v in videos if os.path.splitext(v)[0].lower() in existing]
    toDo = [v for v in videos if os.path.splitext(v)[0].lower() not in existing]

    print(f"Skipped {len(toSkip)} already processed videos.")
    print(f"Resuming from {len(toDo)} remaining videos...\n")

    for fileName in tqdm(toDo, desc=f"Processing {className}"):
        videoPath = os.path.join(classPath, fileName)
        tensorPath = os.path.join(outputDir, f"{className}_{fileName}.pt")

        try:
            frames, _, _ = read_video(videoPath, pts_unit="sec")
        except Exception as e:
            print(f"Skipping corrupted video: {fileName} ({str(e)})")
            continue

        total = frames.shape[0]
        maxStart = max(0, total - (videoFrames - 1) * frameInterval)
        start = random.randint(0, maxStart) if maxStart > 0 else 0

        selected = []
        for i in range(videoFrames):
            frameNum = start + i * frameInterval
            if frameNum < total:
                image = Image.fromarray(frames[frameNum].numpy())
                selected.append(image)
            else:
                break

        while len(selected) < maxFrames:
            selected.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        with torch.no_grad():
            pixelValues = processor(
                images=selected,
                return_tensors="pt",
                do_rescale=False
            )["pixel_values"]

        torch.save({
            "pixel_values": pixelValues.squeeze(0),
            "label": label
        }, tensorPath)
        print(f"Saved {tensorPath}")

# run both classes
for className, label in labelMap.items():
    processAndSave(className, label)

# count final results
print("\n=== Final counts ===")
for className in labelMap:
    count = sum(1 for f in os.listdir(outputDir) if f.startswith(f"{className}_") and f.endswith(".pt"))
    print(f"{className}: {count} tensors")

print("\nPreprocessing complete. Files saved in:", outputDir)