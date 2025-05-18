import os
import cv2
import kagglehub
import numpy as np
from tqdm import tqdm

#config
VIDEO_DIR = os.path.join(kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset"),"Real Life Violence Dataset")
FRAMES_SKIPPED = 2
MOTION_THRESHOLD = 2.0
LABEL_MAP = {"Violence": 1, "NonViolence": 0}


def motionScore(path, FRAMES_SKIPPED):
    cap = cv2.VideoCapture(path)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0

    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motionScores = []

    while True:
        for _ in range(FRAMES_SKIPPED):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motionScores.append(np.mean(mag))
        prevGray = gray

    cap.release()
    return np.mean(motionScores) if motionScores else 0

#classify
def classify_video_by_motion(path, threshold=MOTION_THRESHOLD):
    score = motionScore(path, FRAMES_SKIPPED=FRAMES_SKIPPED)
    return 1 if score >= threshold else 0

#batch evaluate
def evaluate_dir(threshold=MOTION_THRESHOLD):
    preds, labels = [], []
    for labelDir in ["Violence", "NonViolence"]:
        fullPath = os.path.join(VIDEO_DIR, labelDir)
        if not os.path.isdir(fullPath):
            continue
        for fname in tqdm(os.listdir(fullPath), desc=f"Processing {labelDir}"):
            if not fname.endswith((".mp4")):
                continue
            fpath = os.path.join(fullPath, fname)
            pred = classify_video_by_motion(fpath, threshold=threshold) 
            preds.append(pred)
            labels.append(LABEL_MAP[labelDir])
    return preds, labels

#metrics
if __name__ == "__main__":
    from sklearn.metrics import classification_report, confusion_matrix

    preds, labels = evaluate_dir(VIDEO_DIR)
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    print("\nClassification Report:\n", classification_report(labels, preds, target_names=["NonViolence", "Violence"]))
