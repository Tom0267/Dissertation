from transformers import MllamaForConditionalGeneration, AutoProcessor, GenerationConfig
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import textClassifier
from tqdm import tqdm
import torch
import json
import time
import csv
import os

#config
MODELID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
#FRAMEDIR = "RWF-2000/extractedFrames"
FRAMEDIR = "TestedFrames"
SPLITPATH = "RWF-2000/split.json"
RESULTSPATH = "RWF-2000/results.csv"
MAXTOKENS = 200
prompt = (
    "This image is part of a public dataset of street and public scenes used for academic research. "
    "Start your response with a yes or no if violence is depicted in this image. "
    "Then describe what is happening. If there is a violent or aggressive action, explain it and who is involved. "
    "If there isn't any violence, describe the scene as peaceful or non-violent. "
    "Use simple language and avoid complex terms where possible."
)

#load model and processor
model = MllamaForConditionalGeneration.from_pretrained(MODELID, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
model.tie_weights()                                                         
model.eval()
processor = AutoProcessor.from_pretrained(MODELID)                          #load processor for model
model.generation_config = GenerationConfig.from_pretrained(MODELID)         #load generation config from model
model.generation_config.max_new_tokens = MAXTOKENS                          #set max tokens for generation
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id     #set pad token for model
model.generation_config.eos_token_id = processor.tokenizer.eos_token_id     #set generation config for model

class frameDataset(Dataset):
    def __init__(self, frame_paths):    #initialize with a list of frame paths
        self.frame_paths = frame_paths

    def __len__(self):                  #return number of frames
        return len(self.frame_paths)

    def __getitem__(self, idx):         #get a single frame by index
        path = self.frame_paths[idx]
        image = Image.open(path).convert("RGB") #open image and convert to RGB
        return {"image": image, "path": path}
    
#used by DataLoader to combine individual samples into batch.
def collate(batch):
    images = [item["image"] for item in batch]
    paths = [item["path"] for item in batch]
    return {"image": images, "path": paths}
    
def groupFramesByVideo(directory):
    videoGroups = {}
    for fileName in os.listdir(directory):  #list all files in directory
        if not fileName.endswith(".jpg"):   #only process jpg files
            continue
        parts = fileName.split("_") #split by underscore to get video ID and frame number
        if len(parts) < 4:          #ensure enough parts to identify video ID
            continue

        videoId = "_".join(parts[:3])   #join the first three parts as video ID
        if videoId not in videoGroups:  #initialize a new list for this video ID
            videoGroups[videoId] = []   
        videoGroups[videoId].append(os.path.join(directory, fileName))  #add the full path to list
        videoGroups[videoId].sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1])) #sort frames by number
    return videoGroups

def cleanModelResponse(rawText, promptText):
    response = rawText.lower().strip()  #remove whitespace and make lowercase

    #remove roles and common separators
    for role in ["user", "assistant", "system", "llama"]:
        response = response.replace(role + "\n", "")
        response = response.replace(role + ":", "")
        response = response.replace(role, "")
    
    response = response.replace(promptText.lower(), "") #remove prompt if echoed

    response = response.replace("\n", " ").replace("  ", " ").strip()   #remove newlines and extra spaces

    return response
                                                      
def analyseVideoFrames(videoId, framePaths, frameWriter):
    captions = []
    dataset = frameDataset(framePaths)  #create dataset from frame paths
    frameLoader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True, persistent_workers=True) #create DataLoader for batching frames

    #time the image test
    imageStart = time.time()
    batched_results = imageTest(frameLoader)    #run the image test on the batched frames
    imageEnd = time.time()
    imageTime = imageEnd - imageStart

    for path, response in batched_results:  
        refusal = checkBoilerplate(response)        #check for boilerplate refusals
        if refusal is not None:
            response = response + " " + refusal             #append refusal message if found
        label = textClassifier.classify(response)["label"]  #classify the response using text classifier
        captions.append(response)       #append response to captions list

        frameWriter.writerow([              #write frame results to CSV
            videoId,
            os.path.basename(path),
            response,
            label
        ])

        if label == "violent":
            return True, None, response, imageTime  #stop early if violence detected

    return False, captions, None, imageTime 

def imageTest(dataloader):
    results = []
    for batch in dataloader:    #iterate over batches of frames
        images = batch["image"]
        paths = batch["path"]

        messages = [
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            for image in images
        ]
        inputPrompts = processor.apply_chat_template(messages, add_generation_prompt=True)      #apply chat template to format input prompts
        
        nested_images = [[image] for image in images]       #convert images to nested list format for processor 

        inputs = processor(images=nested_images, text=inputPrompts, padding=True, return_tensors="pt").to(model.device) #process inputs for the model

        with torch.no_grad():
            output_ids = model.generate(**inputs)   #generate output IDs from model

        for i in range(len(paths)):     
            decoded = processor.decode(output_ids[i], skip_special_tokens=True).lower() #decode the output IDs to text
            clean = cleanModelResponse(decoded, prompt) #clean response text
            results.append((paths[i], clean))   #add path and cleaned response to results

    return results

def checkBoilerplate(text):
    text = text.lower()
    if "cutting knowledge date" in text or "today date" in text:    #check for specific boilerplate responses
        return "this image was not described due to model refusal."

    elif "i cannot provide" in text or "not appropriate to describe" in text or "i can't help with that request." in text:  #check for general refusal phrases 
        return "Violence was detected but not described due to model refusal."

    return None

if __name__ == "__main__":
    with open(SPLITPATH) as f:  #load the split data to get test videos
        split_data = json.load(f)
        
    testVideos = {vid for vid, split in split_data.items() if split == "test"}  #get test videos from split data
    allFrames = groupFramesByVideo(FRAMEDIR)                                  #group frames by video ID
    groupedFrames = {vid: frames for vid, frames in allFrames.items() if vid in testVideos} #filter frames to only include test videos
    
    with open(RESULTSPATH, "a", newline='', encoding="utf-8", buffering=1) as csvfile, open("frameOutputs.csv", "w", newline='', encoding="utf-8", buffering=1) as framefile:   
        videoWriter = csv.writer(csvfile)
        frameWriter = csv.writer(framefile)

        #headers
        videoWriter.writerow(["VideoID", "PrimaryDecision", "SecondaryDecision", "FinalDecision", "ViolentCaption", "TextOnlyResponse"])
        frameWriter.writerow(["VideoID", "Frame", "RawCaption", "ClassifiedAs"])

        print(f"\nAnalysing {len(groupedFrames)} videos...\n")
        print("VideoID,ViolenceDetected")
        
        totalImageTestTime = 0.0

        for videoId, frames in tqdm(groupedFrames.items(), desc="Analysing videos", unit="video"):      #iterate over each video and its frames
            print(f"\nAnalysing {videoId}...")
            primaryIsViolent, captions, response, imageTime = analyseVideoFrames(videoId, frames, frameWriter)  #analyse the frames of the video
            totalImageTestTime += imageTime
            finalIsViolent = primaryIsViolent   
            secondaryDecision = "N/A"
            textOnlyResponse = "N/A"

            videoWriter.writerow([  #write results to CSV
                videoId,
                "Yes" if primaryIsViolent else "No",
                secondaryDecision,
                "Yes" if primaryIsViolent else "No",
                response if response else "N/A",
                textOnlyResponse if not primaryIsViolent else "N/A"
            ])

            print(f"{videoId} Final: {'Yes' if primaryIsViolent else 'No'}")
        print(f"Total processing time for image test: {totalImageTestTime:.2f} seconds")