import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

MODEL_NAME = "EleutherAI/pythia-410m"

MAX_LENGTH = 128
BATCH_SIZE = 2          # Small batch for CPU
LEARNING_RATE = 1e-5
EPOCHS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# -------------------------------------------------------
# Load Dolly Dataset
# -------------------------------------------------------

dataset = load_dataset(
    "json",
    data_files="databricks-dolly-15k.jsonl"
)

dataset = dataset["train"]

# Use only first 1000 samples (recommended for CPU)
dataset = dataset.select(range(50))

print(dataset)

# -------------------------------------------------------
# Load Tokenizer
# -------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager"
)

model.to(device)

# -------------------------------------------------------
# Convert JSON into Prompt Format
# -------------------------------------------------------

def format_example(example):

    if example["context"]:

        text = (
            f"### Instruction:\n"
            f"{example['instruction']}\n\n"
            f"### Context:\n"
            f"{example['context']}\n\n"
            f"### Response:\n"
            f"{example['response']}"
        )

    else:

        text = (
            f"### Instruction:\n"
            f"{example['instruction']}\n\n"
            f"### Response:\n"
            f"{example['response']}"
        )

    return {"text": text}

dataset = dataset.map(format_example)

# Split dataset into train/validation splits (80/20)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# -------------------------------------------------------
# Tokenization
# -------------------------------------------------------

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    # Decoder models use input_ids as labels. 
    # We replace padding tokens with -100 to ignore them in the loss computation.
    tokens["labels"] = [
        t if t != tokenizer.pad_token_id else -100 for t in tokens["input_ids"]
    ]
    return tokens

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)
val_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

# -------------------------------------------------------
# DataLoader
# -------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------------------------------------------
# Optimizer
# -------------------------------------------------------

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE

)

# -------------------------------------------------------
# Training
# -------------------------------------------------------

print("\nTraining Started\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        if torch.isnan(loss):
            print("\nWarning: Loss is NaN! Skipping backward step.")
            continue

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent NaN / exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        progress.set_description(f"Epoch {epoch+1}")
        progress.set_postfix(loss=loss.item())

    average_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Average Train Loss = {average_train_loss:.4f}")

    # Calculate Validation Loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item()

    average_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} Average Validation Loss = {average_val_loss:.4f}\n")

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------

model.save_pretrained("pythia_dolly")

tokenizer.save_pretrained("pythia_dolly")

print("\nModel Saved")

# -------------------------------------------------------
# Test the Model
# -------------------------------------------------------

model.eval()

prompt = """### Instruction:
Explain Machine Learning.

### Response:
"""

inputs = tokenizer(

    prompt,

    return_tensors="pt"

).to(device)

with torch.no_grad():

    output = model.generate(

        **inputs,

        max_new_tokens=100,

        temperature=0.7,

        top_p=0.9,

        do_sample=True,

        pad_token_id=tokenizer.eos_token_id

    )

generated_text = tokenizer.decode(

    output[0],

    skip_special_tokens=True

)

print("\nGenerated Text\n")

print(generated_text)