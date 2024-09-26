import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp

MODEL_NAME_1 = "EleutherAI/gpt-j-6B"
MODEL_NAME_2 = "EleutherAI/gpt-neox-20b"
JUDGE_MODEL_NAME = "roberta-large"

def load_model_and_tokenizer(model_name, model_class=AutoModelForCausalLM):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

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

def judge_explanation(judge_model, judge_tokenizer, text, explanation):
    inputs = judge_tokenizer(text, explanation, return_tensors="pt", max_length=512, truncation=True).to(judge_model.device)
    with torch.no_grad():
        outputs = judge_model(**inputs)
    return outputs.logits.squeeze()[1].item()

def compete_and_evaluate(agent1, agent1_tokenizer, agent2, agent2_tokenizer, judge_model, judge_tokenizer, text):
    explanation1 = generate_explanation(agent1, agent1_tokenizer, text)
    explanation2 = generate_explanation(agent2, agent2_tokenizer, text)
    
    score1 = judge_explanation(judge_model, judge_tokenizer, text, explanation1)
    score2 = judge_explanation(judge_model, judge_tokenizer, text, explanation2)
    
    if score1 > score2:
        return 1, score1, 0
    elif score2 > score1:
        return 2, 0, score2
    else:
        return 0, 0, 0  # Tie, no reward

def train_agents(agent1, agent1_tokenizer, agent2, agent2_tokenizer, judge_model, judge_tokenizer, texts):
    ppo_model1 = AutoModelForCausalLMWithValueHead.from_pretrained(agent1.config.name_or_path)
    ppo_model2 = AutoModelForCausalLMWithValueHead.from_pretrained(agent2.config.name_or_path)
    
    ppo_config = PPOConfig(
        batch_size=4,
        learning_rate=1e-5,
        ppo_epochs=5,
        gradient_accumulation_steps=4,
    )
    
    ppo_trainer1 = PPOTrainer(ppo_config, ppo_model1, agent1_tokenizer)
    ppo_trainer2 = PPOTrainer(ppo_config, ppo_model2, agent2_tokenizer)
    
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
            query_tensor = agent1_tokenizer.encode(prompt, return_tensors="pt").to(ppo_model1.device)
            
            # Generate explanations and compete
            winner, reward1, reward2 = compete_and_evaluate(ppo_model1, agent1_tokenizer, ppo_model2, agent2_tokenizer, judge_model, judge_tokenizer, text)
            
            # Update only the winning model
            if winner == 1:
                response_tensor1 = ppo_model1.generate(query_tensor)
                ppo_trainer1.step(query_tensor, response_tensor1, torch.tensor([reward1]).to(ppo_model1.device))
            elif winner == 2:
                response_tensor2 = ppo_model2.generate(query_tensor)
                ppo_trainer2.step(query_tensor, response_tensor2, torch.tensor([reward2]).to(ppo_model2.device))
    
    return ppo_trainer1.model, ppo_trainer2.model

def load_and_prepare_data():
    human_texts = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    human_texts = [text for text in human_texts["text"] if len(text.split()) > 50][:5000]
    
    ai_texts = load_dataset("EleutherAI/pile", split="train")
    ai_texts = [text for text in ai_texts["text"] if len(text.split()) > 50][:5000]
    
    return human_texts, ai_texts

def determine_winner(agent1, agent1_tokenizer, agent2, agent2_tokenizer, judge_model, judge_tokenizer, eval_texts):
    agent1_score = 0
    agent2_score = 0
    
    for text in tqdm(eval_texts, desc="Determining winner"):
        winner, _, _ = compete_and_evaluate(agent1, agent1_tokenizer, agent2, agent2_tokenizer, judge_model, judge_tokenizer, text)
        if winner == 1:
            agent1_score += 1
        elif winner == 2:
            agent2_score += 1
    
    if agent1_score > agent2_score:
        return "Agent 1", agent1, agent1_tokenizer
    else:
        return "Agent 2", agent2, agent2_tokenizer

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
    print("Loading models...")
    agent1, agent1_tokenizer = load_model_and_tokenizer(MODEL_NAME_1)
    agent2, agent2_tokenizer = load_model_and_tokenizer(MODEL_NAME_2)
    judge_model, judge_tokenizer = load_model_and_tokenizer(JUDGE_MODEL_NAME, AutoModelForSequenceClassification)

    print("Loading data...")
    human_texts, ai_texts = load_and_prepare_data()
    all_texts = human_texts + ai_texts
    train_texts, eval_texts = train_test_split(all_texts, test_size=0.2)
    
    print("Training agents...")
    improved_agent1, improved_agent2 = train_agents(agent1, agent1_tokenizer, agent2, agent2_tokenizer, judge_model, judge_tokenizer, train_texts)
    
    print("Determining the winner...")
    winner_name, winner_model, winner_tokenizer = determine_winner(
        improved_agent1, agent1_tokenizer, 
        improved_agent2, agent2_tokenizer, 
        judge_model, judge_tokenizer, 
        eval_texts
    )
    print(f"\nThe winner is: {winner_name}")
    
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
