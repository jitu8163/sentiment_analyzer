from fastapi import FastAPI
from app.schema import TextIn, PredictionOut
from app.model import predict_sentiment

app = FastAPI(title="Sentiment Analyzer API")

@app.post("/predict", response_model=PredictionOut)
def predict(data: TextIn):
    label, confidence = predict_sentiment(data.text)
    return {
        "sentiment": label,
        "confidence": round(confidence, 4)
    }
