import os
import cv2
import torch
import json
import random
import kagglehub
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import skew
from datasets import load_dataset, DatasetDict
import torchvision.transforms.functional as F
from torchvision.io import read_video, write_video
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ColorJitter, ToTensor, RandomApply, RandomHorizontalFlip, RandomResizedCrop, RandomRotation

#config
datasetRoot = os.path.join(kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset"),"Real Life Violence Dataset")
#datasetRoot = "RWF-2000/RWF-2000-Dataset/train"
frameOutputDir = "extractedFrames"
featureOutputPath = "motion_features.json"
splitOutputPath = "split.json"
tensorDirs = {
    "train": "train_tensors",
    "val": "val_tensors",
    "test": "test_tensors"
}
numFrames = 16
frameInterval = 15
resizeShape = (224, 224)
valRatio = 0.1
testRatio = 0.2

#preprocessing for each frame 
transform = Compose([   
    Resize((224, 224)),
    RandomApply([ColorJitter(brightness=0.2)], p=0.3),
    RandomApply([ColorJitter(contrast=0.2)], p=0.3),
    RandomApply([ColorJitter(saturation=0.2)], p=0.3),
    RandomApply([ColorJitter(hue=0.1)], p=0.2),
    RandomHorizontalFlip(p=0.5),
    RandomApply([RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))], p=0.2),
    RandomApply([RandomRotation(degrees=10)], p=0.2),
])

class VideoTransform:
    def __init__(self, resize=(224, 224)):      #resize shape for each frame
        self.resize = resize

    def init_random_params(self, image):            #initialize random parameters for augmentation
        self.do_brightness = random.random() < 0.3
        self.do_contrast = random.random() < 0.3
        self.do_saturation = random.random() < 0.3
        self.do_hue = random.random() < 0.2
        self.do_flip = random.random() < 0.5
        self.do_crop = random.random() < 0.2
        self.do_rotate = random.random() < 0.2

        self.brightness_factor = random.uniform(0.8, 1.2) if self.do_brightness else 1.0
        self.contrast_factor = random.uniform(0.8, 1.2) if self.do_contrast else 1.0
        self.saturation_factor = random.uniform(0.8, 1.2) if self.do_saturation else 1.0
        self.hue_factor = random.uniform(-0.1, 0.1) if self.do_hue else 0.0
        self.rotation_angle = random.uniform(-10, 10) if self.do_rotate else 0

        if self.do_crop:
            self.crop_params = RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(1.0, 1.0))  #get random crop parameters
        else:
            self.crop_params = None

    def __call__(self, image):                      #apply transformations to the image 
        img = F.resize(image, self.resize)
        img = F.adjust_brightness(img, self.brightness_factor)
        img = F.adjust_contrast(img, self.contrast_factor)
        img = F.adjust_saturation(img, self.saturation_factor)
        img = F.adjust_hue(img, self.hue_factor)
        if self.do_flip:
            img = F.hflip(img)
        if self.crop_params:
            i, j, h, w = self.crop_params
            img = F.resized_crop(img, i, j, h, w, self.resize)
        if self.do_rotate:
            img = F.rotate(img, self.rotation_angle)
        return img

def extractUniformFrames(videoPath, maxFrames=numFrames, interval=frameInterval):       #extract frames uniformly from the video
    cap = cv2.VideoCapture(videoPath)                   #open video file
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))      #get total number of frames
    frames = []

    for i in range(maxFrames):      #extract frames at regular intervals
        idx = i * interval
        if idx >= total:     #if index exceeds total frames, break
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)   #set the frame position
        ret, frame = cap.read()     #read the frame
        if not ret:                 #if frame not found, break
            break
        frames.append(frame)  #keep as BGR

    cap.release()       #release resource
    return frames

def saveFramesAsJpg(frames, baseName):      #save extracted frames as JPEG images
    os.makedirs(frameOutputDir, exist_ok=True)
    for i, frame in enumerate(frames):
        outPath = os.path.join(frameOutputDir, f"{baseName}_{i}.jpg")
        cv2.imwrite(outPath, frame)

def computeMotionFeatures(frames):      #compute motion features from the extracted frames
    if len(frames) < 2:
        return {k: 0.0 for k in [
            "mean", "std", "max", "min", "range",
            "median", "skew", "frameVar", "motionCount"
        ]}

    motionMeans = []
    motionCounts = []
    prevGray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)  #convert first frame to grayscale

    for frame in frames[1:]:        #iterate over remaining frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #convert to grayscale
        flow = cv2.calcOpticalFlowFarneback(    #calculate optical flow
            prevGray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])    #convert flow to magnitude
        motionMeans.append(np.mean(mag))                        #append mean magnitude to motionMeans
        motionCounts.append(np.sum(mag > 1.0))                  #count significant motion pixels
        prevGray = gray

    motionMeans = np.array(motionMeans)                     #convert to numpy array

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

def saveFramesAsTensor(frames, outPath):    #save frames as a tensor
    tensor = torch.tensor(np.stack(frames), dtype=torch.uint8).permute(0, 3, 1, 2)
    torch.save({"pixel_values": tensor}, outPath)
    
def saveSplit(paths, label_map, split_name, split_map):     #save split information 
    for path in paths:
        label = label_map[path]
        base_name = os.path.splitext(os.path.basename(path))[0]
        tagged_name = f"{label}_{base_name}"
        split_map[tagged_name] = split_name

def preprocessDataset():
    for d in tensorDirs.values():       #create directories for each split
        os.makedirs(d, exist_ok=True)
    os.makedirs(frameOutputDir, exist_ok=True)  #create directory for extracted frames

    videoPaths, labels = [], []

    # use folder names for label assignment
    for labelFolder in os.listdir(datasetRoot):             #iterate over folders
        labelPath = os.path.join(datasetRoot, labelFolder)
        if not os.path.isdir(labelPath):                        #skip if not a directory
            continue
        labelLower = labelFolder.lower()
        if labelLower in ["violence", "fights", "fight"]:
            label = "Violence"
        elif labelLower in ["nonviolence", "nofights", "nonfight"]:
            label = "NonViolence"
        else:
            continue
        for filename in os.listdir(labelPath):                  #iterate over files in label folder
            if filename.endswith((".avi", ".mp4")):             #check for video files
                fullPath = os.path.join(labelPath, filename)
                videoPaths.append(fullPath)
                labels.append(label)

    print(f"Found {len(videoPaths)} video paths.")
    print(f"Label distribution: {dict((label, labels.count(label)) for label in set(labels))}")
    print(f"Sample paths: {videoPaths[:3]}")

    if len(videoPaths) == 0:
        raise RuntimeError("No videos were found.")
    if len(set(labels)) < 2:
        raise RuntimeError("only one class present.")

    labelMap = {path: label for path, label in zip(videoPaths, labels)}     #create map of video paths to labels
    
    
    if testRatio != 1:      #if training
        trainValPaths, testPaths, trainValLabels, testLabels = train_test_split(
            videoPaths, labels, test_size=testRatio, stratify=labels, random_state=42       #split into train+val and test sets
        )

        trainPaths, valPaths, _, _ = train_test_split(
            trainValPaths, trainValLabels, test_size=valRatio / (1.0 - testRatio),      #split into train and val sets
            stratify=trainValLabels, random_state=42
        )

        splitMap = {}
        saveSplit(trainPaths, labelMap, "train", splitMap)
        saveSplit(valPaths, labelMap, "val", splitMap)
        saveSplit(testPaths, labelMap, "test", splitMap)
        
    else:               #if testing
        splitMap = {}
        for path in videoPaths:     #assign all videos to test split
            label = labelMap[path]
            baseName = os.path.splitext(os.path.basename(path))[0]
            taggedName = f"{label}_{baseName}"
            splitMap[taggedName] = "test"
        
    allFeatures = {}
    totalVideos = 0

    for path in tqdm(videoPaths, desc="Processing videos"):     #iterate over all video paths
        label = labelMap[path]
        baseName = os.path.splitext(os.path.basename(path))[0]
        taggedName = f"{label}_{baseName}"
        split = splitMap[taggedName]
        outPath = os.path.join(tensorDirs[split], f"{taggedName}.pt")

        frames = extractUniformFrames(path)
        if len(frames) < 8:
            continue
        
        vt = VideoTransform(resize=resizeShape) #initialize video transform with resize shape
        
        if (split == "train" or split == "val") and random.random() < 0.5:                      #apply augmentations for train and val splits
            vt.init_random_params(Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)))  #initialize random parameters for augmentations
            frames = [
                np.array(vt(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))   #apply transformations to each frame in the video
                for frame in frames
            ]
        else:
            #apply basic resize even if augmentations are skipped
            frames = [
                np.array(F.resize(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), resizeShape))
                for frame in frames
            ]
        saveFramesAsJpg(frames, taggedName)
        saveFramesAsTensor(frames, outPath)
        allFeatures[taggedName] = computeMotionFeatures(frames)
        totalVideos += 1

    with open(featureOutputPath, "w") as f:
        json.dump(allFeatures, f, indent=2)     #save motion features to JSON file
    with open(splitOutputPath, "w") as f:
        json.dump(splitMap, f, indent=2)    #save split map to JSON file

    print(f"\nPreprocessing complete. {totalVideos} videos processed.")
    for split, folder in tensorDirs.items():
        print(f"{split.capitalize()} tensors: {folder}/")
    print(f"Frames saved to: {frameOutputDir}/")
    print(f"Motion features saved to: {featureOutputPath}")
    print(f"Split map saved to: {splitOutputPath}")

if __name__ == "__main__":
    preprocessDataset()