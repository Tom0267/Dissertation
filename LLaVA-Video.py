# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("lmms-lab/LLaVA-Video-7B-Qwen2")
model = AutoModelForCausalLM.from_pretrained("lmms-lab/LLaVA-Video-7B-Qwen2")
model = model.to(device)

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# sentence  = 'Hello World!'
# tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
# model     = BertModel.from_pretrained('bert-large-uncased')

# inputs    = tokenizer(sentence, return_tensors="pt").to(device)
# model     = model.to(device)
# outputs   = model(**inputs)