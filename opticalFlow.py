import os
import cv2
import kagglehub
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score

# config
VIDEO_DIR = os.path.join(kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset"), "Real Life Violence Dataset")
LABEL_MAP = {"Violence": 1, "NonViolence": 0}

# parameters to match other models
NUM_FRAMES = 20
FRAME_INTERVAL = 15
RESIZE_SHAPE = (224, 224)
MOTION_THRESHOLD = 4.5

def sweepThresholds(videoPaths, labels, thresholds=np.arange(0.0, 10.5, 0.5)):
    scores = [motionScore(path) for path in tqdm(videoPaths, desc="Scoring videos")]
    results = []

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]

        f1 = f1_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        accuracy = accuracy_score(labels, preds)

        results.append({
            "threshold": t,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        })

    # Best threshold by F1
    best = max(results, key=lambda x: x["f1"])
    print(f"\n📌 Best Threshold (F1): {best['threshold']:.2f}")
    print(f"   F1: {best['f1']:.4f} | Precision: {best['precision']:.4f} | Recall: {best['recall']:.4f} | Accuracy: {best['accuracy']:.4f}")

    plotThresholdSweep(results)
    return best, results


def plotThresholdSweep(results):
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in results]
    f1s = [r["f1"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1s, label="F1", marker='o')
    plt.plot(thresholds, precisions, label="Precision", marker='x')
    plt.plot(thresholds, recalls, label="Recall", marker='s')
    plt.plot(thresholds, accuracies, label="Accuracy", linestyle='--')
    plt.xlabel("Motion Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep: 0.0 to 10.0")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def extractUniformFrames(videoPath, numFrames=NUM_FRAMES, frameInterval=FRAME_INTERVAL):
    cap = cv2.VideoCapture(videoPath)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = []

    for i in range(numFrames):
        frameIdx = i * frameInterval
        if frameIdx >= totalFrames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, RESIZE_SHAPE)
        selected.append(frame)

    cap.release()
    return selected

def motionScore(videoPath):
    frames = extractUniformFrames(videoPath)

    if len(frames) < 2:
        return 0.0  # not enough frames to compare

    motionStats = []
    prevGray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prevGray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        motionStats.append({
            "mean": float(np.mean(mag)),
            "std": float(np.std(mag)),
            "max": float(np.max(mag))
        })

        prevGray = gray

    if not motionStats:
        return 0.0

    return np.mean([m["mean"] for m in motionStats])

def classifyByMotion(path, threshold=MOTION_THRESHOLD):
    score = motionScore(path)
    return 1 if score >= threshold else 0

def evaluate(threshold=MOTION_THRESHOLD):
    preds, labels = [], []
    for labelDir in ["Violence", "NonViolence"]:
        fullPath = os.path.join(VIDEO_DIR, labelDir)
        if not os.path.isdir(fullPath):
            continue
        for fname in tqdm(os.listdir(fullPath), desc=f"Processing {labelDir}"):
            if not fname.endswith(".mp4"):
                continue
            fpath = os.path.join(fullPath, fname)
            pred = classifyByMotion(fpath, threshold=threshold)
            preds.append(pred)
            labels.append(LABEL_MAP[labelDir])
    return preds, labels

if __name__ == "__main__":
    videoPaths = []
    labels = []

    for labelDir in ["Violence", "NonViolence"]:
        fullPath = os.path.join(VIDEO_DIR, labelDir)
        for fname in os.listdir(fullPath):
            if fname.endswith(".mp4"):
                videoPaths.append(os.path.join(fullPath, fname))
                labels.append(LABEL_MAP[labelDir])

    #best_threshold, _ = sweepThresholds(videoPaths, labels)

    preds, labels = evaluate()

    print("confusion matrix: " + str(confusion_matrix(labels, preds)))
    print("classification report: " + str(classification_report(labels, preds)))
    print("accuracy: " + str(accuracy_score(labels, preds)))
    print("f1 score: " + str(f1_score(labels, preds)))
    print("precision: " + str(precision_score(labels, preds)))
    print("recall: " + str(recall_score(labels, preds)))
    print("roc_auc: " + str(roc_auc_score(labels, preds)))