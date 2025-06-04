from transformers import MllamaForConditionalGeneration, AutoProcessor, GenerationConfig
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import textClassifier
from tqdm import tqdm
import kagglehub
import torch
import time
import csv
import os
import cv2

#config
modelId = "meta-llama/Llama-3.2-11B-Vision-Instruct"
frameDirectory = "extractedFrames"
prompt = (
    "This image is part of a public dataset of street and public scenes used for academic research. "
    "Start your response with a yes or no if violence is depicted in this image. "
    "Then describe what is happening. If there is a violent or aggressive action, explain it and who is involved. "
    "If there isn't any violence, describe the scene as peaceful or non-violent. "
    "Use simple language and avoid complex terms where possible."
)
maxTokens = 300
resultsPath = "results.csv"

#load model and processor
model = MllamaForConditionalGeneration.from_pretrained(modelId, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
model.tie_weights()
model.eval()
processor = AutoProcessor.from_pretrained(modelId)
model.generation_config = GenerationConfig.from_pretrained(modelId)
model.generation_config.max_new_tokens = maxTokens
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
model.generation_config.eos_token_id = processor.tokenizer.eos_token_id

class frameDataset(Dataset):
    def __init__(self, frame_paths):
        self.frame_paths = frame_paths

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        image = Image.open(path).convert("RGB")
        return {"image": image, "path": path}
    
def collate(batch):
    images = [item["image"] for item in batch]
    paths = [item["path"] for item in batch]
    return {"image": images, "path": paths}
    
def groupFramesByVideo(directory):
    videoGroups = {}
    for fileName in os.listdir(directory):
        if not fileName.endswith(".jpg"):
            continue
        parts = fileName.split("_")
        if len(parts) < 4:
            continue
        # e.g. Violence_V_1_0 → Violence_V_1
        videoId = "_".join(parts[:3])
        if videoId not in videoGroups:
            videoGroups[videoId] = []
        videoGroups[videoId].append(os.path.join(directory, fileName))
        videoGroups[videoId].sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
    return videoGroups

def cleanModelResponse(rawText, promptText):
    response = rawText.lower().strip()

    #remove known roles and common separators
    for role in ["user", "assistant", "system", "llama"]:
        response = response.replace(role + "\n", "")
        response = response.replace(role + ":", "")
        response = response.replace(role, "")

    #remove prompt if echoed
    response = response.replace(promptText.lower(), "")

    #replace multiple line breaks or stray \n
    response = response.replace("\n", " ").replace("  ", " ").strip()

    return response

# def checkViolenceInFrame(framePath):
#     image = Image.open(framePath).convert("RGB")
#     messages = [
#         {"role": "user", "content": [
#             {"type": "image"},
#             {"type": "text", "text": prompt}
#         ]}
#     ]
#     inputText = processor.apply_chat_template(messages, add_generation_prompt=True)
#     inputs = processor(image, inputText, add_special_tokens=False, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputIds = model.generate(**inputs, max_new_tokens=maxTokens)
#         result = processor.decode(outputIds[0], skip_special_tokens=True).lower()
#         result = cleanModelResponse(result, prompt)

#     print(f"{os.path.basename(framePath)} → {result}\n")
#     return  result
                                                      
def analyseVideoFrames(videoId, framePaths, frameWriter):
    captions = []
    dataset = frameDataset(framePaths)
    frameLoader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True, persistent_workers=True)

    batched_results = imageTest(frameLoader)

    for path, response in batched_results:
        refusal = checkBoilerplate(response)
        if refusal is not None:
            response = response + " " + refusal
        label = textClassifier.classify(response)["label"]
        captions.append(response)

        frameWriter.writerow([
            videoId,
            os.path.basename(path),
            response,
            label
        ])

        if label == "violent":
            return True, None, response  # stop early if violence detected

    return False, captions, None

def imageTest(dataloader):
    results = []
    for batch in dataloader:
        images = batch["image"]
        paths = batch["path"]

        messages = [
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            for image in images
        ]
        inputPrompts = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # this is key — nesting the images
        nested_images = [[image] for image in images]

        inputs = processor(images=nested_images, text=inputPrompts, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs)

        for i in range(len(paths)):
            decoded = processor.decode(output_ids[i], skip_special_tokens=True).lower()
            clean = cleanModelResponse(decoded, prompt)
            results.append((paths[i], clean))

    return results

def checkBoilerplate(text):
    text = text.lower()
    if "cutting knowledge date" in text or "today date" in text:
        return "this image was not described due to model refusal."

    elif "i cannot provide" in text or "not appropriate to describe" in text or "i can't help with that request." in text:
        return "Violence was detected but not described due to model refusal."

    return None

if __name__ == "__main__":
    groupedFrames = groupFramesByVideo(frameDirectory)
    with open(resultsPath, "a", newline='', encoding="utf-8", buffering=1) as csvfile, open("frameOutputs.csv", "w", newline='', encoding="utf-8", buffering=1) as framefile:
        videoWriter = csv.writer(csvfile)
        frameWriter = csv.writer(framefile)

        # headers
        videoWriter.writerow(["VideoID", "PrimaryDecision", "SecondaryDecision", "FinalDecision", "ViolentCaption", "TextOnlyResponse"])
        frameWriter.writerow(["VideoID", "Frame", "RawCaption", "ClassifiedAs"])

        print(f"\nAnalysing {len(groupedFrames)} videos...\n")
        print("VideoID,ViolenceDetected")
        
        startTime = time.time()
        for videoId, frames in tqdm(groupedFrames.items(), desc="Analysing videos", unit="video"):
            print(f"\nAnalysing {videoId}...")
            primaryIsViolent, captions, response = analyseVideoFrames(videoId, frames, frameWriter)
            finalIsViolent = primaryIsViolent
            secondaryDecision = "N/A"
            textOnlyResponse = "N/A"

            if not primaryIsViolent and captions:
                combinedText = " ".join(captions)
                secondPrompt = (
                    "Based on the following scene descriptions, decide if this video is violent. "
                    "Respond with 'yes' or 'no'. Descriptions: " + combinedText
                )
                messages = [{"role": "user", "content": [{"type": "text", "text": secondPrompt}]}]
                inputText = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=inputText, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputIds = model.generate(**inputs, max_new_tokens=50)
                    textOnlyResponse = processor.decode(outputIds[0], skip_special_tokens=True).lower()
                    textOnlyResponse = cleanModelResponse(textOnlyResponse, secondPrompt)

                print(f"Text-only response: {textOnlyResponse.strip()}")
                secondaryDecision = "yes" if textClassifier.classify(textOnlyResponse)["label"] == "violent" else "No"
                finalIsViolent = (secondaryDecision == "yes")

            videoWriter.writerow([
                videoId,
                "Yes" if primaryIsViolent else "No",
                secondaryDecision,
                "Yes" if finalIsViolent else "No",
                response if response else "N/A",
                textOnlyResponse if not primaryIsViolent else "N/A"
            ])

            print(f"{videoId} Final: {'Yes' if finalIsViolent else 'No'}")
        endTime = time.time()
        print(f"Processing time for {videoId}: {endTime - startTime:.2f} seconds")