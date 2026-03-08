import streamlit as st
import pickle
import string
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    L = []
    for word in text:
        if word.isalnum():
            L.append(word)

    text = L[:]
    L.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            L.append(word)

    text = L[:]
    L.clear()

    for word in text:
        L.append(ps.stem(word))

    return " ".join(L)


# Load model and vectorizer safely
BASE_DIR = os.path.dirname(__file__)

vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

tfidf = pickle.load(open(vectorizer_path, "rb"))
model = pickle.load(open(model_path, "rb"))


# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

st.write("This AI model detects whether a message is **Spam** or **Not Spam** using Natural Language Processing.")

input_sms = st.text_area("Enter the message")


if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:

        # 1. Preprocess text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
