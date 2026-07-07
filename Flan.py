import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.eval()


text = """
The Apollo 11 mission was the first manned mission to land on the Moon. 
Launched by NASA in July 1969, the spacecraft carried astronauts Neil Armstrong, 
Buzz Aldrin, and Michael Collins. On July 20, Armstrong and Aldrin successfully 
landed the Apollo Lunar Module on the lunar surface, and Armstrong became 
the first person to step onto the Moon.
"""

# FLAN-T5 performs tasks using instructions
prompt = "Summarize: " + text


inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True
)



with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=4,
        early_stopping=True
    )

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input Text:\n")
print(text)

print("\nSummary:\n")
print(summary)