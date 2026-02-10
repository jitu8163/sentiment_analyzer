import re
import pandas as pd

def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df = pd.read_csv("/home/ubuntu/Desktop/sentiment analysis/data/reviews.csv")
df["review"] = df["review"].apply(clean_text)
df = df.head(3500)

df.to_csv("data/clean_reviews.csv", index=False)
print("Cleaning done. Saved as clean_reviews.csv")
