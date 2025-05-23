from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import skew
from tqdm import tqdm
import numpy as np
import kagglehub
import cv2
import os

#config
VIDEO_DIR = os.path.join(kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset"), "Real Life Violence Dataset")
LABEL_MAP = {"Violence": 1, "NonViolence": 0}

#parameters to match other models
NUM_FRAMES = 20
FRAME_INTERVAL = 15
RESIZE_SHAPE = (224, 224)
MOTION_THRESHOLD = 4.5

def sweepFeatureThreshold(videoPaths, labels, feature, thresholds=np.arange(0.0, 10, 0.5)):
    print(f"\nSweeping feature: {feature}")
    
    #extract chosen feature for each video
    featureValues = [motionFeatures(path)[feature] for path in tqdm(videoPaths, desc=f"Extracting '{feature}'")]

    results = []
    for t in thresholds:
        preds = [1 if val >= t else 0 for val in featureValues]

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

    #best threshold for that feature
    best = max(results, key=lambda x: x["f1"])
    print(f"Best for '{feature}': {best['threshold']:.2f} | F1: {best['f1']:.4f} | Precision: {best['precision']:.4f} | Recall: {best['recall']:.4f} | Accuracy: {best['accuracy']:.4f}")

    return results, best

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

def motionFeatures(videoPath):
    frames = extractUniformFrames(videoPath)

    if len(frames) < 2:
        return {
            "mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "range": 0.0,
            "median": 0.0, "skew": 0.0, "frame_var": 0.0, "motion_count": 0
        }

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
        motionCounts.append(np.sum(mag > 1.0))  #number of active pixels

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
        "frame_var": float(np.var(motionMeans)),
        "motion_count": int(np.mean(motionCounts))
    }

def classifyByMotion(path, threshold=MOTION_THRESHOLD):
    score = motionFeatures(path)["mean"]
    return 1 if score >= threshold else 0

def evaluate(threshold=MOTION_THRESHOLD):
    preds, labels, scores = [], [], []
    for labelDir in ["Violence", "NonViolence"]:
        fullPath = os.path.join(VIDEO_DIR, labelDir)
        if not os.path.isdir(fullPath):
            continue
        for fname in tqdm(os.listdir(fullPath), desc=f"Processing {labelDir}"):
            if not fname.endswith(".mp4"):
                continue
            fpath = os.path.join(fullPath, fname)
            features = motionFeatures(fpath)
            score = features["mean"]
            pred = 1 if score >= threshold else 0

            preds.append(pred)
            scores.append(score)
            labels.append(LABEL_MAP[labelDir])

    return preds, labels, scores

if __name__ == "__main__":
    videoPaths = []
    labels = []

    for labelDir in ["Violence", "NonViolence"]:
        fullPath = os.path.join(VIDEO_DIR, labelDir)
        for fname in os.listdir(fullPath):
            if fname.endswith(".mp4"):
                videoPaths.append(os.path.join(fullPath, fname))
                labels.append(LABEL_MAP[labelDir])

    for feat in ["mean", "std", "range", "motion_count"]:
        sweepFeatureThreshold(videoPaths, labels, feature=feat)

    preds, labels, scores = evaluate()

    print("confusion matrix: " + str(confusion_matrix(labels, preds)))
    print("classification report: " + str(classification_report(labels, preds)))
    print("accuracy: " + str(accuracy_score(labels, preds)))
    print("f1 score: " + str(f1_score(labels, preds)))
    print("precision: " + str(precision_score(labels, preds)))
    print("recall: " + str(recall_score(labels, preds)))
    print("roc_auc: " + str(roc_auc_score(labels, scores)))