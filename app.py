import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
import numpy as np



nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()


@st.cache_resource
def load_trained_model():
    return joblib.load('modelo_do_grupo.joblib')

trained_model = load_trained_model()




st.title("Classificador de Sentimento de Texto")


user_input = st.text_area("Digite o texto para classificação:")

if st.button("Classificar"):
    if user_input:
        processed_text = preprocess_text(user_input)
        
        embedding = embedding_model.encode([processed_text])
        
        prediction = trained_model.predict(embedding)
        prediction_proba = trained_model.predict_proba(embedding)
        
        sentiment = "Positivo" if prediction[0] == 1 else "Negativo"
        confidence = np.max(prediction_proba) * 100
        
        st.write(f"**Sentimento previsto:** {sentiment}")
        st.write(f"**Confiança:** {confidence:.2f}%")
    else:
        st.warning("Por favor, insira um texto para classificação.")
