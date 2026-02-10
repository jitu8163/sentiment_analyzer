import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

labels = ["negative", "positive"]

def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    idx = torch.argmax(probs).item()
    return labels[idx], float(probs[0][idx])
