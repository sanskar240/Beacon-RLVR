from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Load and preprocess SFT dataset
dataset = load_dataset("csv", data_files="data/sft_dataset.csv")["train"]
def tokenize_fn(examples):
    return tokenizer(
        [instr + " " + resp for instr, resp in zip(examples["instruction"], examples["response"])],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
dataset = dataset.map(tokenize_fn, batched=True)

# Training
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="models/sft",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        fp16=True,
        save_steps=500,
        logging_steps=100,
        learning_rate=2e-4,
        save_strategy="steps",
        logging_strategy="steps"
    ),
    train_dataset=dataset
)

trainer.train()
model.save_pretrained("models/sft")
tokenizer.save_pretrained("models/sft")