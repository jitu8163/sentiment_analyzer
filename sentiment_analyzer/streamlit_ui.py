import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🎬 Sentiment Analyzer")
st.write("Enter a movie review and get sentiment using fine-tuned DistilBERT.")

text = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            result = response.json()
            st.success(f"Sentiment: **{result['sentiment'].upper()}**")
            st.info(f"Confidence: {result['confidence']:.2f}")
        else:
            st.error("Error connecting to API")
