from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_name = "moka-ai/m3e-base"
cache_dir = "./models/m3e-base"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
