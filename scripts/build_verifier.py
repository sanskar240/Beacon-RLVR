from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from collections import Counter
import torch

# Load preference pairs
dataset = load_dataset("csv", data_files="/content/beacon_rlvr/data/beacon-verifier.csv")["train"]

# Extract sycophantic keywords
sycophantic_words = []
for ex in dataset:
    sycophantic_words.extend(ex["sycophantic_response"].lower().split())
keywords = [w for w, c in Counter(sycophantic_words).most_common(20) if w not in ["the", "is", "a", "and", "to", "of", "in"]]
print("Sycophantic keywords:", keywords)

# Rule-based scorer
def rule_scorer(text):
    score = 1.0
    for word in keywords:
        if word in text.lower():
            score -= 0.2
    for phrase in ["maybe", "perhaps", "it seems", "possibly", "kind of", "sort of"]:
        if phrase in text.lower():
            score -= 0.1
    brevity = max(0, 1 - len(text.split()) / 50)
    score += 0.2 * brevity
    return max(0.0, min(1.0, score))

# Prepare classifier data
def make_pairs(examples):
    return {
        "text": examples["blunt_response"] + examples["sycophantic_response"],
        "label": [1] * len(examples["blunt_response"]) + [0] * len(examples["sycophantic_response"])
    }

classifier_data = dataset.map(make_pairs, batched=True, remove_columns=dataset.column_names)
classifier_data = classifier_data.train_test_split(test_size=0.2)

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized = classifier_data.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="/content/beacon_rlvr/models/classifier",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=True
    ),
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"]
)

print("Training classifier...")
trainer.train()

# Save
model.save_pretrained("/content/beacon_rlvr/models/classifier")
tokenizer.save_pretrained("/content/beacon_rlvr/models/classifier")

# Hybrid verifier
from transformers import pipeline
classifier = pipeline("text-classification", model="/content/beacon_rlvr/models/classifier", tokenizer=tokenizer, device=0)

def hybrid_verifier(response, prompt=""):
    rule_score = rule_scorer(response)
    try:
        cls_out = classifier(response)[0]
        cls_score = cls_out["score"] if cls_out["label"] == "LABEL_1" else 1 - cls_out["score"]
    except:
        cls_score = 0.5
    return 0.7 * rule_score + 0.3 * cls_score

# Validate
held_out = dataset.select(range(336, 420))
correct = sum(hybrid_verifier(ex["blunt_response"]) > hybrid_verifier(ex["sycophantic_response"]) for ex in held_out)
print(f"Verifier Accuracy: {correct}/{len(held_out)} = {correct/len(held_out):.1%}")