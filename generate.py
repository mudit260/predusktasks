from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Force CPU
device = torch.device("cpu")

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)
model.eval()

# Take prompt input
prompt = input("Enter your prompt: ")

# Encode input
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Function to generate text
def generate_with_temperature(input_ids, temp):
    print(f"\nGenerating with temperature = {temp}...")
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate with temp = 0.7
output_07 = generate_with_temperature(input_ids, temp=0.7)

# Generate with temp = 1.0
output_10 = generate_with_temperature(input_ids, temp=1.0)

# Print outputs
print("\n=== Output with Temperature 0.7 ===\n")
print(output_07)

print("\n=== Output with Temperature 1.0 ===\n")
print(output_10)

# Save to files
with open("output_temp_0.7.txt", "w", encoding="utf-8") as f:
    f.write(output_07)

with open("output_temp_1.0.txt", "w", encoding="utf-8") as f:
    f.write(output_10)
