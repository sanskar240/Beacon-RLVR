import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
print(f"CUDA Available: {torch.cuda.is_available()}")
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
dataset = load_dataset("csv", data_files="data/beacon-verifier.csv")
print("Environment ready!")