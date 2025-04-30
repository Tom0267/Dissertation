# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("lmms-lab/LLaVA-Video-7B-Qwen2")
model = AutoModelForCausalLM.from_pretrained("lmms-lab/LLaVA-Video-7B-Qwen2")