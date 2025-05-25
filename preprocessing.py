import os
import cv2
import torch
import json
import kagglehub
import numpy as np
from tqdm import tqdm
from scipy.stats import skew

#config
datasetRoot = os.path.join(kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset"),"Real Life Violence Dataset")
frameOutputDir = "extractedFrames"
featureOutputPath = "motion_features.json"
tensorOutputDir = "test_videos"

numFrames = 20
frameInterval = 15
resizeShape = (224, 224)

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
        frame = cv2.resize(frame, resizeShape)
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
    torch.save(tensor, outPath)

def preprocessDataset():
    os.makedirs(tensorOutputDir, exist_ok=True)
    os.makedirs(frameOutputDir, exist_ok=True)
    allFeatures = {}
    totalVideos = 0

    for label in ["Violence", "NonViolence"]:
        fullPath = os.path.join(datasetRoot, label)
        for fileName in tqdm(os.listdir(fullPath), desc=f"Processing {label}"):
            if not fileName.endswith(".mp4"):
                continue

            videoPath = os.path.join(fullPath, fileName)
            baseName = os.path.splitext(fileName)[0]
            taggedName = f"{label}_{baseName}"
            ptPath = os.path.join(tensorOutputDir, f"{taggedName}.pt")

            if os.path.exists(ptPath):
                continue

            #extract frames once
            frames = extractUniformFrames(videoPath)
            if len(frames) < 2:
                continue

            #use exact same frames for all outputs
            saveFramesAsJpg(frames, taggedName)             #for LLaVA
            saveFramesAsTensor(frames, ptPath)              #for Timesformer
            allFeatures[videoPath] = computeMotionFeatures(frames)  #for Random Forest
            totalVideos += 1

    with open(featureOutputPath, "w") as f:
        json.dump(allFeatures, f, indent=2)

    print(f"\nPreprocessing complete. {totalVideos} test videos processed.")
    print(f"Frames saved to: {frameOutputDir}/")
    print(f"Features saved to: {featureOutputPath}")
    print(f"Tensors saved to: {tensorOutputDir}/")

if __name__ == "__main__":
    preprocessDataset()