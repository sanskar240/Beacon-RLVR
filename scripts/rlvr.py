import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.distributions import Categorical
from tqdm import tqdm

# Load SFT model and classifier
model = AutoModelForCausalLM.from_pretrained("models/sft", torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("models/sft")
classifier_pipe = pipeline("text-classification", model="models/classifier", tokenizer="distilbert-base-uncased")

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Load prompts
dataset = load_dataset("csv", data_files="data/beacon-verifier.csv")["train"]
rl_prompts = dataset.map(lambda ex: {"prompt": ex["prompt"]}).select(range(84))

# Hybrid verifier
def hybrid_verifier(response, prompt):
    sycophantic_keywords = ["amazing", "fantastic", "wonderful", "great", "awesome"]
    hedging_phrases = ["maybe", "perhaps", "it seems", "possibly"]
    rule_score = 1.0
    for word in sycophantic_keywords:
        if word in response.lower():
            rule_score -= 0.2
    for phrase in hedging_phrases:
        if phrase in response.lower():
            rule_score -= 0.1
    word_count = len(response.split())
    brevity_bonus = max(0, 1 - word_count / 50)
    rule_score += 0.2 * brevity_bonus
    rule_score = max(0, min(1, rule_score))
    semantic_score = classifier_pipe(response)[0]
    semantic_score = semantic_score["score"] if semantic_score["label"] == "LABEL_1" else 1 - semantic_score["score"]
    return 0.7 * rule_score + 0.3 * semantic_score

# RLVR loop
num_epochs = 3
num_samples_per_prompt = 4
baseline = 0.5

for epoch in range(num_epochs):
    total_loss = 0
    total_reward = 0
    for example in tqdm(rl_prompts, desc=f"Epoch {epoch+1}"):
        prompt = example["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to("cuda")

        responses = []
        log_probs = []
        for _ in range(num_samples_per_prompt):
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True
            )
            response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            responses.append(response)

            scores = torch.stack(outputs.scores, dim=1)
            probs = torch.softmax(scores, dim=-1)
            token_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
            log_prob = torch.log(probs.gather(2, token_ids.unsqueeze(-1))).mean()
            log_probs.append(log_prob)

        rewards = [hybrid_verifier(resp, prompt) for resp in responses]
        total_reward += sum(rewards) / len(rewards)
        baseline = 0.9 * baseline + 0.1 * (sum(rewards) / len(rewards))

        for log_prob, reward in zip(log_probs, rewards):
            advantage = reward - baseline
            loss = -log_prob * advantage
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

    print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(rl_prompts)}, Avg Reward = {total_reward / len(rl_prompts)}")

model.save_pretrained("models/rlvr")
tokenizer.save_pretrained("models/rlvr")