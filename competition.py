import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os

# Use larger, more powerful models
MODEL_NAME_1 = "EleutherAI/gpt-j-6B"
MODEL_NAME_2 = "EleutherAI/gpt-neox-20b"
JUDGE_MODEL_NAME = "roberta-large"

def load_model_and_tokenizer(model_name, model_class=AutoModelForCausalLM):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

agent1_model, agent1_tokenizer = load_model_and_tokenizer(MODEL_NAME_1)
agent2_model, agent2_tokenizer = load_model_and_tokenizer(MODEL_NAME_2)
judge_model, judge_tokenizer = load_model_and_tokenizer(JUDGE_MODEL_NAME, AutoModelForSequenceClassification)

def generate_explanation(model, tokenizer, text, max_length=500):
    prompt = f"""Analyze if the following text is human-written or AI-generated. Provide a detailed explanation following this structure:
1. Initial assessment
2. Analysis of language complexity and style
3. Evaluation of content coherence and depth
4. Identification of any unusual patterns or inconsistencies
5. Conclusion with confidence level

Text: "{text}"

Explanation:"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def fine_tune_judge_model(judge_model, judge_tokenizer, train_data, eval_data):
    class JudgeDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text, explanation, label = self.data[idx]
            inputs = self.tokenizer(text, explanation, truncation=True, max_length=512, padding="max_length")
            return {
                "input_ids": torch.tensor(inputs["input_ids"]),
                "attention_mask": torch.tensor(inputs["attention_mask"]),
                "labels": torch.tensor([label])
            }

    train_dataset = JudgeDataset(train_data, judge_tokenizer)
    eval_dataset = JudgeDataset(eval_data, judge_tokenizer)

    training_args = TrainingArguments(
        output_dir="./judge_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=judge_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    return trainer.model

def judge_explanation(judge_model, judge_tokenizer, text, explanation):
    inputs = judge_tokenizer(text, explanation, return_tensors="pt", max_length=512, truncation=True).to(judge_model.device)
    with torch.no_grad():
        outputs = judge_model(**inputs)
    return outputs.logits.squeeze()[1].item()  # Return the score for the positive class

def run_competition(agent1_model, agent1_tokenizer, agent2_model, agent2_tokenizer, judge_model, judge_tokenizer, text):
    explanation1 = generate_explanation(agent1_model, agent1_tokenizer, text)
    explanation2 = generate_explanation(agent2_model, agent2_tokenizer, text)
    
    score1 = judge_explanation(judge_model, judge_tokenizer, text, explanation1)
    score2 = judge_explanation(judge_model, judge_tokenizer, text, explanation2)
    
    return explanation1, explanation2, score1, score2

def train_agent(agent_model, agent_tokenizer, texts, judge_model, judge_tokenizer):
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(agent_model.config.name_or_path)
    ref_model = create_reference_model(ppo_model)
    
    ppo_config = PPOConfig(
        batch_size=4,
        learning_rate=1e-5,
        ppo_epochs=5,
        gradient_accumulation_steps=4,
    )
    
    ppo_trainer = PPOTrainer(ppo_config, ppo_model, ref_model, agent_tokenizer)
    
    for epoch in range(ppo_config.ppo_epochs):
        for text in tqdm(texts, desc=f"Training epoch {epoch+1}"):
            prompt = f"""Analyze if the following text is human-written or AI-generated. Provide a detailed explanation following this structure:
1. Initial assessment
2. Analysis of language complexity and style
3. Evaluation of content coherence and depth
4. Identification of any unusual patterns or inconsistencies
5. Conclusion with confidence level

Text: "{text}"

Explanation:"""
            query_tensor = agent_tokenizer.encode(prompt, return_tensors="pt").to(ppo_model.device)
            response_tensor = ppo_model.generate(query_tensor)
            explanation = agent_tokenizer.decode(response_tensor[0])
            
            reward = judge_explanation(judge_model, judge_tokenizer, text, explanation)
            reward_tensor = torch.tensor([reward]).to(ppo_model.device)
            
            stats = ppo_trainer.step(query_tensor, response_tensor, reward_tensor)
    
    return ppo_trainer.model

def load_and_prepare_data():
    human_texts = load_dataset("wikipedia", "20220301.en", split="train")
    human_texts = [text for text in human_texts["text"] if len(text.split()) > 50][:5000]
    
    ai_texts = load_dataset("EleutherAI/pile", split="train")
    ai_texts = [text for text in ai_texts["text"] if len(text.split()) > 50][:5000]
    
    return human_texts, ai_texts


def determine_winner(agent1_model, agent1_tokenizer, agent2_model, agent2_tokenizer, judge_model, judge_tokenizer, eval_texts):
    agent1_score = 0
    agent2_score = 0
    
    for text in tqdm(eval_texts, desc="Determining winner"):
        explanation1, explanation2, score1, score2 = run_competition(
            agent1_model, agent1_tokenizer, 
            agent2_model, agent2_tokenizer, 
            judge_model, judge_tokenizer, 
            text
        )
        
        if score1 > score2:
            agent1_score += 1
        elif score2 > score1:
            agent2_score += 1
    
    if agent1_score > agent2_score:
        return "Agent 1", agent1_model, agent1_tokenizer
    else:
        return "Agent 2", agent2_model, agent2_tokenizer
def save_model(model, tokenizer, name):
    output_dir = f"./{name}_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def load_saved_model(name):
    model_dir = f"./{name}_model"
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def user_interface(model, tokenizer):
    while True:
        user_input = input("Enter a text to analyze (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        explanation = generate_explanation(model, tokenizer, user_input)
        print("\nExplanation:")
        print(explanation)
        print("\n" + "="*50 + "\n")

def main():
    print("Loading data...")
    human_texts, ai_texts = load_and_prepare_data()
    all_texts = human_texts + ai_texts
    train_texts, eval_texts = train_test_split(all_texts, test_size=0.2)
    
    # Prepare data for fine-tuning the judge model
    train_explanations = [generate_explanation(agent1_model, agent1_tokenizer, text) for text in tqdm(train_texts[:1000], desc="Generating train explanations")]
    eval_explanations = [generate_explanation(agent1_model, agent1_tokenizer, text) for text in tqdm(eval_texts[:200], desc="Generating eval explanations")]
    
    train_data = list(zip(train_texts[:1000], train_explanations, [1]*500 + [0]*500))  # Assume first 500 are human, last 500 are AI
    eval_data = list(zip(eval_texts[:200], eval_explanations, [1]*100 + [0]*100))
    
    print("Fine-tuning judge model...")
    judge_model = fine_tune_judge_model(judge_model, judge_tokenizer, train_data, eval_data)
    
    print("Training Agent 1...")
    improved_agent1 = train_agent(agent1_model, agent1_tokenizer, train_texts, judge_model, judge_tokenizer)
    
    print("Training Agent 2...")
    improved_agent2 = train_agent(agent2_model, agent2_tokenizer, train_texts, judge_model, judge_tokenizer)
    
    print("\nEvaluating improved agents...")
    for i, text in enumerate(eval_texts[:5]):
        explanation1, explanation2, score1, score2 = run_competition(
            improved_agent1, agent1_tokenizer, 
            improved_agent2, agent2_tokenizer, 
            judge_model, judge_tokenizer, 
            text
        )
        
        print(f"\nSample {i+1}:")
        print(f"Text: {text[:100]}...")
        print(f"\nAgent 1 Explanation (Score: {score1:.2f}):\n{explanation1}")
        print(f"\nAgent 2 Explanation (Score: {score2:.2f}):\n{explanation2}")
        print("\n" + "="*50)
    
    print("Determining the winner...")
    winner_name, winner_model, winner_tokenizer = determine_winner(
        improved_agent1, agent1_tokenizer, 
        improved_agent2, agent2_tokenizer, 
        judge_model, judge_tokenizer, 
        eval_texts
    )
    print(f"\nThe winner is: {winner_name}")
    
    print("Saving the winning model...")
    save_model(winner_model, winner_tokenizer, "winning_agent")
    
    print("\nYou can now use the winning model to analyze texts.")
    user_interface(winner_model, winner_tokenizer)
    
    
if __name__ == "__main__":
    if os.path.exists("./winning_agent_model"):
        print("Loading saved winning model...")
        model, tokenizer = load_saved_model("winning_agent")
        user_interface(model, tokenizer)
    else:
        main()
