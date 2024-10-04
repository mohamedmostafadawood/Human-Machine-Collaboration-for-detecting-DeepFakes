import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_saved_model(name="winning_agent"):
    model_dir = f"./{name}_model"
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def generate_explanation(model, tokenizer, text, max_new_tokens=200):
    device = next(model.parameters()).device
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

def user_interface(model, tokenizer):
    print("Welcome to the Text Analysis Tool!")
    print("This tool will analyze if a given text is likely human-written or AI-generated.")
    print("Enter 'quit' to exit the program.")
    print("\n" + "="*50 + "\n")

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

    print("Thank you for using the Text Analysis Tool!")

# Load the saved model
model, tokenizer = load_saved_model()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Start the user interface
user_interface(model, tokenizer)
