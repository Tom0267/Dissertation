import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

#config
modelId = "meta-llama/Llama-3.2-11B-Vision-Instruct"
frameDirectory = "extractedFrames"
prompt = "Does this image show a violent incident?"

#load model and processor
model = MllamaForConditionalGeneration.from_pretrained(
    modelId,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.tie_weights()
model.eval()
processor = AutoProcessor.from_pretrained(modelId)

#group frames by video number
def groupFramesByVideo(directory):
    videoGroups = {}
    for fileName in os.listdir(directory):
        if not fileName.endswith(".jpg"):
            continue
        parts = fileName.split("_")
        if len(parts) < 3:
            continue
        videoId = parts[1]  #e.g. NV_1_0_X.jpg → videoId = "1"
        if videoId not in videoGroups:
            videoGroups[videoId] = []
        videoGroups[videoId].append(os.path.join(directory, fileName))
    return videoGroups

def checkViolenceInFrame(framePath):
    image = Image.open(framePath).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    inputText = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, inputText, add_special_tokens=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputIds = model.generate(**inputs, max_new_tokens=30)
        result = processor.decode(outputIds[0], skip_special_tokens=True).lower()

    print(f"{os.path.basename(framePath)} → {result}")
    return any(term in result for term in ["yes", "violent", "fight", "aggression", "physical"])

#analyse all frames in one video
def analyseVideoFrames(framePaths):
    for path in framePaths:
        if checkViolenceInFrame(path):
            return True  #flag video as violent
    return False  #no frame triggered a violence flag

if __name__ == "__main__":
    groupedFrames = groupFramesByVideo(frameDirectory)
    print(f"\nAnalysing {len(groupedFrames)} videos...\n")

    print("VideoID,ViolenceDetected")
    for videoId, frames in groupedFrames.items():
        frames.sort()  #ensure consistent order
        isViolent = analyseVideoFrames(frames)
        print(f"{videoId},{'Yes' if isViolent else 'No'}\n")