import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "philschmid/tiny-bert-sst2-distilled"

BATCH_SIZE = 32
LEARNING_RATE = 3e-5
EPOCHS = 4
MAX_LENGTH = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# ----------------------------------------------------------
# Load IMDb Dataset
# ----------------------------------------------------------

print("\nLoading IMDb dataset...")

dataset = load_dataset("csv", data_files="IMDB Dataset.csv")

# Split dataset into train and test splits (80/20)
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

print(split_dataset)

# ----------------------------------------------------------
# Load Tokenizer
# ----------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ----------------------------------------------------------
# Tokenization Function
# ----------------------------------------------------------

def tokenize(batch):
    # Tokenize the reviews
    inputs = tokenizer(
        batch["review"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    # Map 'positive'/'negative' to 1/0
    inputs["label"] = [1 if s == "positive" else 0 for s in batch["sentiment"]]
    return inputs

# Tokenize entire dataset
tokenized_dataset = split_dataset.map(
    tokenize,
    batched=True
)

# Keep only required columns
tokenized_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

# ----------------------------------------------------------
# DataLoaders
# ----------------------------------------------------------

train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    tokenized_dataset["test"],
    batch_size=BATCH_SIZE
)

# ----------------------------------------------------------
# Load Pretrained TinyBERT
# ----------------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(device)

# ----------------------------------------------------------
# Optimizer
# ----------------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------

print("\nStarting Training...\n")

model.train()

for epoch in range(EPOCHS):

    total_loss = 0

    progress_bar = tqdm(train_loader)

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_description(
            f"Epoch {epoch+1}"
        )

        progress_bar.set_postfix(
            loss=loss.item()
        )

    average_loss = total_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Average Loss: {average_loss:.4f}")

# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------

print("\nEvaluating Model...\n")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in tqdm(test_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

SAVE_PATH = f"tinybert_lr{LEARNING_RATE}_bs{BATCH_SIZE}_ep{EPOCHS}_wd0.01"

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"Model saved to {SAVE_PATH}")

# ----------------------------------------------------------
# Test on New Sentences
# ----------------------------------------------------------

model.eval()

sentences = [

    "The movie was fantastic and I loved every minute.",

    "This was the worst movie I have ever watched.",

    "The acting was average but the story was interesting.",

    "I would never recommend this film."

]

print("\nPredictions:\n")

for sentence in sentences:

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    label = "Positive" if prediction == 1 else "Negative"

    print(f"Sentence : {sentence}")
    print(f"Prediction : {label}")
    print("-" * 50)
