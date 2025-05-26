import os
import cv2
import torch
import json
import kagglehub
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ColorJitter, ToTensor, RandomApply

#config
datasetRoot = os.path.join(
    kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset"),
    "Real Life Violence Dataset"
)
frameOutputDir = "extractedFrames"
featureOutputPath = "motion_features.json"
splitOutputPath = "split.json"
tensorDirs = {
    "train": "train_tensors",
    "val": "val_tensors",
    "test": "test_tensors"
}
numFrames = 20
frameInterval = 15
resizeShape = (224, 224)
valRatio = 0.1
testRatio = 0.1

#preprocessing for each frame
transform = Compose([
    Resize((224, 224)),
    RandomApply([ColorJitter(brightness=0.2)], p=0.2),
    RandomApply([ColorJitter(contrast=0.2)], p=0.2),
    RandomApply([ColorJitter(saturation=0.2)], p=0.2),
    RandomApply([ColorJitter(hue=0.1)], p=0.2)
])

def extractUniformFrames(videoPath, maxFrames=numFrames, interval=frameInterval):
    cap = cv2.VideoCapture(videoPath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(maxFrames):
        idx = i * interval
        if idx >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = np.array(transform(image))
        frames.append(frame)

    cap.release()
    return frames

def saveFramesAsJpg(frames, baseName):
    os.makedirs(frameOutputDir, exist_ok=True)
    for i, frame in enumerate(frames):
        outPath = os.path.join(frameOutputDir, f"{baseName}_{i}.jpg")
        cv2.imwrite(outPath, frame)

def computeMotionFeatures(frames):
    if len(frames) < 2:
        return {k: 0.0 for k in [
            "mean", "std", "max", "min", "range",
            "median", "skew", "frameVar", "motionCount"
        ]}

    motionMeans = []
    motionCounts = []
    prevGray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prevGray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motionMeans.append(np.mean(mag))
        motionCounts.append(np.sum(mag > 1.0))
        prevGray = gray

    motionMeans = np.array(motionMeans)

    return {
        "mean": float(np.mean(motionMeans)),
        "std": float(np.std(motionMeans)),
        "max": float(np.max(motionMeans)),
        "min": float(np.min(motionMeans)),
        "range": float(np.max(motionMeans) - np.min(motionMeans)),
        "median": float(np.median(motionMeans)),
        "skew": float(skew(motionMeans)),
        "frameVar": float(np.var(motionMeans)),
        "motionCount": int(np.mean(motionCounts))
    }

def saveFramesAsTensor(frames, outPath):
    tensor = torch.tensor(np.stack(frames), dtype=torch.uint8).permute(0, 3, 1, 2)
    torch.save({"pixel_values": tensor}, outPath)

def preprocessDataset():
    for d in tensorDirs.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(frameOutputDir, exist_ok=True)

    videoPaths, labels = [], []

    for label in ["Violence", "NonViolence"]:
        fullPath = os.path.join(datasetRoot, label)
        for fileName in os.listdir(fullPath):
            if fileName.endswith(".mp4"):
                videoPaths.append(os.path.join(fullPath, fileName))
                labels.append(label)

    #split off test set
    trainValPaths, testPaths, trainValLabels, testLabels = train_test_split(
        videoPaths, labels, test_size=testRatio, stratify=labels, random_state=42
    )

    #split train into train and val
    trainPaths, valPaths, _, _ = train_test_split(
        trainValPaths, trainValLabels, test_size=valRatio / (1.0 - testRatio),
        stratify=trainValLabels, random_state=42
    )

    splitMap = {}
    for path in trainPaths:
        splitMap[path] = "train"
    for path in valPaths:
        splitMap[path] = "val"
    for path in testPaths:
        splitMap[path] = "test"

    allFeatures = {}
    totalVideos = 0

    for path in tqdm(videoPaths, desc="Processing videos"):
        label = os.path.basename(os.path.dirname(path))
        baseName = os.path.splitext(os.path.basename(path))[0]
        taggedName = f"{label}_{baseName}"
        split = splitMap[path]
        outPath = os.path.join(tensorDirs[split], f"{taggedName}.pt")

        frames = extractUniformFrames(path)
        if len(frames) < 2:
            continue
        if split == "test":
            saveFramesAsJpg(frames, taggedName)
        saveFramesAsTensor(frames, outPath)
        allFeatures[path] = computeMotionFeatures(frames)
        totalVideos += 1

    with open(featureOutputPath, "w") as f:
        json.dump(allFeatures, f, indent=2)

    with open(splitOutputPath, "w") as f:
        json.dump(splitMap, f, indent=2)

    print(f"\nPreprocessing complete. {totalVideos} videos processed.")
    for split, folder in tensorDirs.items():
        print(f"{split.capitalize()} tensors: {folder}/")
    print(f"Frames saved to: {frameOutputDir}/")
    print(f"Motion features saved to: {featureOutputPath}")
    print(f"Split map saved to: {splitOutputPath}")

if __name__ == "__main__":
    preprocessDataset()