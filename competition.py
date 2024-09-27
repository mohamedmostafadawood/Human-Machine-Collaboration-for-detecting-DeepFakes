import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import random 
from torch.utils.data import DataLoader

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_device(model):
    return next(model.parameters()).device

# Use smaller models
MODEL_NAME_1 = "gpt2"  # Much smaller than GPT-J-6B
MODEL_NAME_2 = "EleutherAI/gpt-neo-125M"  # Much smaller than GPT-NeoX-20B
JUDGE_MODEL_NAME = "distilroberta-base"  # Smaller than RoBERTa-large

from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, RobertaTokenizer, RobertaForSequenceClassification

def load_model_and_tokenizer(model_name, model_class=AutoModelForCausalLM):
    print(f"Loading {model_name}...")
    device = get_device()
    if 'gpt2' in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    elif 'gpt-neo' in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    elif 'roberta' in model_name.lower():
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name).to(device)
    
    if model is None or tokenizer is None:
        raise ValueError(f"Failed to load model or tokenizer for {model_name}")

    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer
    
def generate_explanation(model, tokenizer, text, max_new_tokens=100):
    device = get_model_device(model)
    prompt = f"Analyze if this text is human-written or AI-generated: '{text}'. Explanation:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    explanation = generated_text[len(prompt):].strip()
    return explanation

def judge_explanation(judge_model, judge_tokenizer, text, explanation):
    device = get_model_device(judge_model)
    inputs = judge_tokenizer(text, explanation, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = judge_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities[0][1].item()

def compete_and_evaluate(agent1, agent1_tokenizer, agent2, agent2_tokenizer, judge_model, judge_tokenizer, text):
    explanation1 = generate_explanation(agent1, agent1_tokenizer, text, max_new_tokens=200)
    explanation2 = generate_explanation(agent2, agent2_tokenizer, text, max_new_tokens=200)
    
    score1 = judge_explanation(judge_model, judge_tokenizer, text, explanation1)
    score2 = judge_explanation(judge_model, judge_tokenizer, text, explanation2)
    
    if score1 > score2:
        return 1, score1, 0
    elif score2 > score1:
        return 2, 0, score2
    else:
        return 0, 0, 0  # Tie, no reward



def train_agents(agent1, agent1_tokenizer, agent2, agent2_tokenizer, judge_model, judge_tokenizer, texts):
    device = get_device()
    ppo_model1 = AutoModelForCausalLMWithValueHead.from_pretrained(agent1.config.name_or_path).to(device)
    ppo_model2 = AutoModelForCausalLMWithValueHead.from_pretrained(agent2.config.name_or_path).to(device)
    
    if agent1_tokenizer is None or agent2_tokenizer is None:
        raise ValueError("Tokenizers are not properly initialized.")
    
    # PPO config for training
    ppo_config = PPOConfig(
        batch_size=128,  # Larger batch size for efficiency, but will adjust dynamically
        mini_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        ppo_epochs=3,
    )
    
    ppo_trainer1 = PPOTrainer(config=ppo_config, model=ppo_model1, tokenizer=agent1_tokenizer)
    ppo_trainer2 = PPOTrainer(config=ppo_config, model=ppo_model2, tokenizer=agent2_tokenizer)
    
    # Create DataLoader for batching, don't drop last incomplete batch
    dataloader = DataLoader(texts, batch_size=ppo_config.batch_size, shuffle=True, drop_last=False)
    
    for epoch in range(ppo_config.ppo_epochs):
        for batch in tqdm(dataloader, desc=f"Training epoch {epoch+1}"):
            queries1, queries2 = [], []
            responses1, responses2 = [], []
            rewards1, rewards2 = [], []
            
            for text in batch:
                prompt = f"Analyze if this text is human-written or AI-generated: '{text}'. Explanation:"
                query_tensor1 = agent1_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                query_tensor2 = agent2_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                
                # Compete agents and determine winner
                winner, reward1, reward2 = compete_and_evaluate(ppo_model1, agent1_tokenizer, ppo_model2, agent2_tokenizer, judge_model, judge_tokenizer, text)
                
                # Append queries, responses, and rewards based on the winner
                if winner == 1:
                    response_tensor1 = ppo_model1.generate(**query_tensor1, max_new_tokens=100)
                    queries1.append(query_tensor1.input_ids[0])
                    responses1.append(response_tensor1[0])
                    rewards1.append(torch.tensor(reward1, device=device))
                elif winner == 2:
                    response_tensor2 = ppo_model2.generate(**query_tensor2, max_new_tokens=100)
                    queries2.append(query_tensor2.input_ids[0])
                    responses2.append(response_tensor2[0])
                    rewards2.append(torch.tensor(reward2, device=device))
            
            # Adjust the batch size to match the actual number of queries for each agent
            actual_batch_size = len(queries1)
            if actual_batch_size > 0:
                print(f"Agent 1 batch sizes: queries={len(queries1)}, responses={len(responses1)}, rewards={len(rewards1)}")
                ppo_trainer1.config.batch_size = actual_batch_size  # Dynamically adjust the batch size
                ppo_trainer1.step(queries1, responses1, rewards1)

            actual_batch_size = len(queries2)
            if actual_batch_size > 0:
                print(f"Agent 2 batch sizes: queries={len(queries2)}, responses={len(responses2)}, rewards={len(rewards2)}")
                ppo_trainer2.config.batch_size = actual_batch_size  # Dynamically adjust the batch size
                ppo_trainer2.step(queries2, responses2, rewards2)
    
    return ppo_trainer1.model, ppo_trainer2.model

    
def load_and_prepare_data():
    # Load a smaller dataset
    dataset = load_dataset("tweet_eval", "sentiment", split="train", trust_remote_code=True)
    texts = dataset["text"]
    
    # Limit to 1000 samples
    texts = texts[:1000]
    
    return texts

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
    print(f"Model and tokenizer saved to {output_dir}")

def load_saved_model(name):
    model_dir = f"./{name}_model"
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def user_interface(model, tokenizer):
    while True:
        user_input = input("Enter a text to analyze (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        try:
            explanation = generate_explanation(model, tokenizer, user_input)
            print("\nExplanation:")
            print(explanation)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        
        print("\n" + "="*50 + "\n")

def main():
    print("Loading models...")
    agent1, agent1_tokenizer = load_model_and_tokenizer(MODEL_NAME_1)
    agent2, agent2_tokenizer = load_model_and_tokenizer(MODEL_NAME_2)
    judge_model, judge_tokenizer = load_model_and_tokenizer(JUDGE_MODEL_NAME, AutoModelForSequenceClassification)

    print(f"Agent1 tokenizer: {type(agent1_tokenizer)}")
    print(f"Agent2 tokenizer: {type(agent2_tokenizer)}")
    print(f"Judge tokenizer: {type(judge_tokenizer)}")

    if agent1_tokenizer is None or agent2_tokenizer is None or judge_tokenizer is None:
        raise ValueError("One or more tokenizers failed to initialize.")
    
    
    print("Loading data...")
    texts = load_and_prepare_data()
    train_texts, eval_texts = train_test_split(texts, test_size=0.2)
    
    print("Training agents...")
    improved_agent1, improved_agent2 = train_agents(agent1, agent1_tokenizer, agent2, agent2_tokenizer, judge_model, judge_tokenizer, train_texts)    
    
    device = get_device()
    improved_agent1 = improved_agent1.to(device)
    improved_agent2 = improved_agent2.to(device)
    
    
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
