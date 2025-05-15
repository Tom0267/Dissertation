from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import textClassifier
import torch
import csv
import os

#config
modelId = "meta-llama/Llama-3.2-11B-Vision-Instruct"
frameDirectory = "extractedFrames"
prompt ="This image is part of a public dataset of street and public scenes used for academic research. start your response with a yes or no if violence depicted in this image then describe what is happening in this image, If there is a violent or aggressive action, explain what it is and who is involved, If there isnt any violence, describe the scene as peaceful or non-violent, use simple language and avoid complex terms where possible. "
maxTokens = 300
resultsPath = "results.csv"

#load model and processor
model = MllamaForConditionalGeneration.from_pretrained(modelId, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
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
        videoId = parts[0] + "_" + parts[1]  #e.g. NV_1_0_X.jpg → videoId = "NV_1"
        if videoId not in videoGroups:
            videoGroups[videoId] = []
        videoGroups[videoId].append(os.path.join(directory, fileName))
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
        outputIds = model.generate(**inputs, max_new_tokens=maxTokens)
        result = processor.decode(outputIds[0], skip_special_tokens=True).lower()
        result = cleanModelResponse(result, prompt)

    print(f"{os.path.basename(framePath)} → {result}\n")
    return  result

#analyse all frames in one video
def analyseVideoFrames(framePaths):
    captions = []
    for path in framePaths:
        response = checkViolenceInFrame(path)
        refusal = checkBoilerplate(response)
        if refusal is not None:
            response = response + refusal
        if textClassifier.classify(response)["label"] == "violent":
            return True, None, response  #flag video as violent
        captions.append(response)
    return False, captions, None  #no frame triggered a violence flag

def checkBoilerplate(text):
    text = text.lower()
    if "cutting knowledge date" in text or "today date" in text:
        return "this image was not described due to model refusal."
    
    elif"i cannot provide" in text or "not appropriate to describe" in text or "i can't help with that request." in text:
        return "Violence was detected but not described due to model refusal."

    return None

if __name__ == "__main__":
    groupedFrames = groupFramesByVideo(frameDirectory)
    with open(resultsPath, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["VideoID", "PrimaryDecision", "SecondaryDecision", "FinalDecision", "ViolentCaption", "TextOnlyResponse"])
                
    print(f"\nAnalysing {len(groupedFrames)} videos...\n")

    print("VideoID,ViolenceDetected")
    for videoId, frames in groupedFrames.items():
            print(f"\nAnalysing {videoId}...")
            frames.sort()
            primaryIsViolent, captions, response = analyseVideoFrames(frames)
            finalIsViolent = primaryIsViolent
            secondaryDecision = "N/A"
            textOnlyResponse = "N/A"

            # text-only fallback if needed
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

                print(f"Secondary check: {textOnlyResponse.strip()}")
                secondaryDecision = "yes" if textClassifier.classify(textOnlyResponse)["label"] == "violent" else "No"

                finalIsViolent = True if secondaryDecision == "yes" else False
                    
            with open(resultsPath, "a", newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    videoId,
                    "Yes" if primaryIsViolent else "No",
                    secondaryDecision,
                    "Yes" if finalIsViolent else "No",
                    response if response else "N/A",
                    textOnlyResponse if not primaryIsViolent else "N/A"
                ])
            print(f"{videoId} Final: {'Yes' if finalIsViolent else 'No'}")            