from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import cv2
import os

#config
VIDEO_DIR = os.path.join(
    kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset"),
    "Real Life Violence Dataset"
)
LABEL_MAP = {"Violence": 1, "NonViolence": 0}
NUM_FRAMES = 20
FRAME_INTERVAL = 15
RESIZE_SHAPE = (224, 224)

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
        return {k: 0.0 for k in [
            "mean", "std", "max", "min", "range",
            "median", "skew", "frame_var", "motion_count"
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
        "frame_var": float(np.var(motionMeans)),
        "motion_count": int(np.mean(motionCounts))
    }

if __name__ == "__main__":
    videoPaths, labels = [], []
    for labelDir in ["Violence", "NonViolence"]:
        fullPath = os.path.join(VIDEO_DIR, labelDir)
        for fname in os.listdir(fullPath):
            if fname.endswith(".mp4"):
                videoPaths.append(os.path.join(fullPath, fname))
                labels.append(LABEL_MAP[labelDir])

    features_by_path = {}
    for path in tqdm(videoPaths, desc="Extracting motion features"):
        features_by_path[path] = motionFeatures(path)

    #format for training
    X = [list(f.values()) for f in features_by_path.values()]
    y = labels
    feature_names = list(next(iter(features_by_path.values())).keys())

    # extract test video paths based on .pt files in test_videos/
test_video_paths = []
y_test, X_test = [], []
X_train, y_train = [], []

for ptFile in os.listdir("test_videos"):
    if not ptFile.endswith(".pt"):
        continue

    baseName = ptFile.removesuffix(".mp4.pt")
    if baseName.startswith("NonViolence_"):
        realName = baseName.replace("NonViolence_", "")
        className = "NonViolence"
    elif baseName.startswith("Violence_"):
        realName = baseName.replace("Violence_", "")
        className = "Violence"
    else:
        continue

    videoName = realName + ".mp4"
    videoPath = os.path.join(VIDEO_DIR, className, videoName)
    test_video_paths.append(videoPath)

    # use test_video_paths to split features
    test_video_set = set(test_video_paths)
    for path, features in features_by_path.items():
        feat_vec = list(features.values())
        label = LABEL_MAP[os.path.basename(os.path.dirname(path))]

        if path in test_video_set:
            X_test.append(feat_vec)
            y_test.append(label)
        else:
            X_train.append(feat_vec)
            y_train.append(label)

    """ define parameter grid
    param_grid = {
        "n_estimators": [40, 50, 60, 70],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_split": [2, 5, 7],
        "min_samples_leaf": [1, 2, 3, 4],
        "class_weight": [None, "balanced"]
    }

    # configure grid search
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    # fit search
    print("\n=== Running Grid Search ===")
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)
    """
    
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n=== Random Forest Evaluation ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    #plot feature importance
    importances = clf.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()

    #plot ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve - Random Forest")
    plt.tight_layout()
    plt.show()