import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------
# Load Pretrained TinyBERT
# ---------------------------------------------------------

model_name = "philschmid/tiny-bert-sst2-distilled"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Binary classification (Positive / Negative)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

model.eval()



sentences = [
    "The movie was absolutely amazing.",
    "The food was terrible.",
    "The service was excellent.",
    "I am very disappointed.",
    
]


for sentence in sentences:

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    label = "Positive" if prediction == 1 else "Negative"

    print(f"\nSentence : {sentence}")
    print(f"Prediction : {label}")