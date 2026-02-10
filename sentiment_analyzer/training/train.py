import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load data
df = pd.read_csv("data/clean_reviews.csv")

label_map = {"negative": 0, "positive": 1}
df["label"] = df["sentiment"].map(label_map)

texts = df["review"].tolist()
labels = df["label"].tolist()

# Use DistilBERT (faster)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Proper split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings   = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset   = ReviewDataset(val_encodings, val_labels)

args = TrainingArguments(
    output_dir="sentiment_model",
    per_device_train_batch_size=16,
    num_train_epochs=1,     
    save_strategy="epoch",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

print("Model saved.")
