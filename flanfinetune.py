import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "google/flan-t5-small"
BATCH_SIZE = 4          # Batch size for CPU training (FLAN-T5-small has ~60M params)
LEARNING_RATE = 5e-5
EPOCHS = 3
MAX_INPUT_LENGTH = 512  # Articles are longer texts
MAX_TARGET_LENGTH = 128 # Summaries are shorter texts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ----------------------------------------------------------
# Load BBC News Summary Dataset
# ----------------------------------------------------------

print("\nLoading BBC News Summary dataset...")
dataset = load_dataset("gopalkalpande/bbc-news-summary")
dataset = dataset["train"]

# Select subset to make training feasible on CPU (e.g. 100 samples)
# Change this number if you have GPU acceleration
dataset_subset = dataset.select(range(100))

# Split into Train and Validation splits (80/20)
split_dataset = dataset_subset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(split_dataset)

# ----------------------------------------------------------
# Load Tokenizer & Model
# ----------------------------------------------------------

print(f"\nLoading model and tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)

# ----------------------------------------------------------
# Tokenization / Preprocessing
# ----------------------------------------------------------

def preprocess_function(examples):
    # Prepend the instruction task prefix for FLAN-T5
    inputs = ["summarize: " + article for article in examples["Articles"]]
    
    # Tokenize the input articles (no padding here; DataCollator handles it dynamically)
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )
    
    # Tokenize the target summaries (no padding here)
    labels = tokenizer(
        text_target=examples["Summaries"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Process dataset splits
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Remove raw text/metadata columns so the DataLoader/DataCollator only processes model inputs
columns_to_remove = ["File_path", "Articles", "Summaries"]
tokenized_train = tokenized_train.remove_columns(columns_to_remove)
tokenized_val = tokenized_val.remove_columns(columns_to_remove)

# ----------------------------------------------------------
# DataCollator (Automatically pads inputs and labels to the batch's max length 
# and masks label padding with -100, converting outputs to PyTorch tensors)
# ----------------------------------------------------------

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    return_tensors="pt"
)

# ----------------------------------------------------------
# DataLoaders
# ----------------------------------------------------------

train_loader = DataLoader(
    tokenized_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)

val_loader = DataLoader(
    tokenized_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)

# ----------------------------------------------------------
# Optimizer
# ----------------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# ----------------------------------------------------------
# Training & Validation Loop
# ----------------------------------------------------------

print("\nStarting Training...\n")

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
            print("\nWarning: Loss is NaN! Skipping step.")
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

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

SAVE_PATH = f"flan_t5_bbc_summarization"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\nModel and tokenizer saved successfully to '{SAVE_PATH}'")

# ----------------------------------------------------------
# Test the Fine-Tuned Model
# ----------------------------------------------------------

print("\nTesting Model on a Sample Article...")
model.eval()

# Select one article from validation set to summarize
sample_article = val_dataset[0]["Articles"]
sample_summary = val_dataset[0]["Summaries"]

print(f"\nOriginal Article excerpt:\n{sample_article[:300]}...")
print(f"\nGround Truth Summary:\n{sample_summary}")

inputs = tokenizer(
    "summarize: " + sample_article,
    return_tensors="pt",
    max_length=MAX_INPUT_LENGTH,
    truncation=True
).to(device)

with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        early_stopping=True
    )

generated_summary = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True
)

print(f"\nGenerated Summary:\n{generated_summary}\n")
