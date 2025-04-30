from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
import torch

#gpu = 0 if torch.cuda.is_available() else -1
gpu = None
if gpu == 0:
    print("Using GPU")
else:
    print("Using CPU")

pipe = pipeline("image-text-to-text", model="meta-llama/Llama-3.2-11B-Vision-Instruct", device=gpu)

messages = [
    {"role": "user", "content": "Who are you?"}
]
output = pipe(messages)

# manually loading (if needed)
#processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
#model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct").to("cuda" if torch.cuda.is_available() else "cpu")