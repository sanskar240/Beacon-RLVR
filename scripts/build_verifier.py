from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from collections import Counter
import re

# Load preference pairs
dataset = load_dataset("csv", data_files="data/beacon-verifier.csv")["train"]

# Extract sycophantic patterns
sycophantic_words = []
for example in dataset:
    sycophantic_words.extend(example["sycophantic_response"].lower().split())
sycophantic_keywords = [word for word, count in Counter(sycophantic_words).most_common(20) if word not in ["the", "is", "a"]]
print("Sycophantic Keywords:", sycophantic_keywords)

# Rule-based verifier
def rule_based_verifier(response, prompt):
    score = 1.0
    for word in sycophantic_keywords:
        if word in response.lower():
            score -= 0.2
    hedging_phrases = ["maybe", "perhaps", "it seems", "possibly"]
    for phrase in hedging_phrases:
        if phrase in response.lower():
            score -= 0.1
    word_count = len(response.split())
    brevity_bonus = max(0, 1 - word_count / 50)
    score += 0.2 * brevity_bonus
    return max(0, min(1, score))

# Train classifier
def format_classifier_data(examples):
    return [
        {"text": examples["blunt_response"], "label": 1, "prompt": examples["prompt"]},
        {"text": examples["sycophantic_response"], "label": 0, "prompt": examples["prompt"]}
    ]
classifier_data = dataset.map(format_classifier_data, batched=True, remove_columns=["blunt_response", "sycophantic_response"]).flatten()
classifier_data = classifier_data.train_test_split(test_size=0.2)

classifier_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_classifier(examples):
    return classifier_tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
classifier_data = classifier_data.map(tokenize_classifier, batched=True)

classifier = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
trainer = Trainer(
    model=classifier,
    args=TrainingArguments(
        output_dir="models/classifier",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    ),
    train_dataset=classifier_data["train"],
    eval_dataset=classifier_data["test"]
)
trainer.train()
classifier.save_pretrained("models/classifier")
classifier_tokenizer.save_pretrained("models/classifier")

# Hybrid verifier
from transformers import pipeline
classifier_pipe = pipeline("text-classification", model="models/classifier", tokenizer=classifier_tokenizer)
def hybrid_verifier(response, prompt):
    rule_score = rule_based_verifier(response, prompt)
    semantic_score = classifier_pipe(response)[0]
    semantic_score = semantic_score["score"] if semantic_score["label"] == "LABEL_1" else 1 - semantic_score["score"]
    return 0.7 * rule_score + 0.3 * semantic_score

# Validate
held_out = dataset.select(range(336, 420))
correct = 0
for example in held_out:
    blunt_score = hybrid_verifier(example["blunt_response"], example["prompt"])
    sycophantic_score = hybrid_verifier(example["sycophantic_response"], example["prompt"])
    if blunt_score > sycophantic_score:
        correct += 1
print(f"Verifier Accuracy: {correct / len(held_out)}")