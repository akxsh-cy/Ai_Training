import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Load Pretrained Pythia Model
# ---------------------------------------------------------

model_name = "EleutherAI/pythia-410m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Set model to evaluation mode
model.eval()

# ---------------------------------------------------------
# Generate Text
# ---------------------------------------------------------

prompt = "the cat didnot get any food so it was"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Prompt:")
print(prompt)

print("\nGenerated Text:")
print(generated_text)