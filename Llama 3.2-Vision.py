from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
import kagglehub
import torch
import cv2
import os

#config
frameDirectory = "extractedFrames"
frameInterval = 30

path = kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset")
videoDirectory = os.path.join(path, "Real Life Violence Dataset")
os.makedirs(frameDirectory, exist_ok=True)


def extractFrames(videoPath, label, outputDirectory, interval=30):
    videoName = os.path.splitext(os.path.basename(videoPath))[0]
    capture = cv2.VideoCapture(videoPath)
    frameCount = 0
    savedCount = 0

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        if frameCount % interval == 0:
            outputName = f"{videoName}_{savedCount}_{label}.jpg"
            outputPath = os.path.join(outputDirectory, outputName)
            cv2.imwrite(outputPath, frame)
            savedCount += 1
        frameCount += 1

    capture.release()
    print(f"Extracted {savedCount} frames from {videoName}.")


for label in ["Violence", "NonViolence"]:
    labelPath = os.path.join(videoDirectory, label)
    for filename in os.listdir(labelPath):
        if filename.endswith(".mp4"):
            fullPath = os.path.join(labelPath, filename)
            extractFrames(fullPath, label, frameDirectory, interval=frameInterval)

print("extracted frames saved to:", frameDirectory)
            

gpu = 0 if torch.cuda.is_available() else -1
gpu = None
if gpu == 0:
    print("Using GPU")
else:
    print("Using CPU")

pipe = pipeline("image-text-to-text", model="meta-llama/Llama-3.2-11B-Vision-Instruct", device=gpu)

messages = [
    {"role": "user", "content": "give me a number between 1 and 10?"}
]
output = pipe(messages)