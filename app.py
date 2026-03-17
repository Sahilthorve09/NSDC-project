# -------------------- IMPORT LIBRARIES --------------------

import pandas as pd
import numpy as np
import streamlit as st
import docx2txt
import pdfplumber
import re
import nltk
import en_core_web_sm
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from spacy.matcher import Matcher

# -------------------- DOWNLOAD NLTK DATA (ONLY IF NOT AVAILABLE) --------------------

def download_nltk():
    resources = ['punkt','wordnet','stopwords','omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)

download_nltk()

stop = set(stopwords.words('english'))

# -------------------- LOAD SPACY MODEL --------------------

nlp = en_core_web_sm.load()
matcher = Matcher(nlp.vocab)

# -------------------- STREAMLIT TITLE --------------------

st.title('📄 RESUME CLASSIFICATION SYSTEM')
st.subheader('Upload Resume to Predict Profile & Extract Skills')

# -------------------- LOAD MODEL & VECTORIZER --------------------

model = load("ModelRFC.joblib")
Vectorizer = load("VECTOR.joblib")

# -------------------- TEXT EXTRACTION --------------------

def extract_text(file):

    text = ""

    if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(file)

    elif file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()

    return text

# -------------------- PREPROCESSING --------------------

def preprocess(sentence):

    sentence = sentence.lower()
    sentence = re.sub('<.*?>', '', sentence)
    sentence = re.sub(r'http\S+', '', sentence)
    sentence = re.sub('[0-9]+', '', sentence)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)

    filtered = [w for w in tokens if len(w) > 2 and w not in stop]

    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(w) for w in filtered]

    return " ".join(lemma)

# -------------------- SKILL EXTRACTION --------------------

def extract_skills(resume_text):

    nlp_text = nlp(resume_text)

    tokens = [token.text for token in nlp_text if not token.is_stop]

    data = pd.read_csv("skills.csv")

    skills_list = [skill.lower() for skill in data.columns.values]

    skillset = []

    for token in tokens:
        if token.lower() in skills_list:
            skillset.append(token)

    for chunk in nlp_text.noun_chunks:
        chunk = chunk.text.lower().strip()
        if chunk in skills_list:
            skillset.append(chunk)

    return list(set(skillset))

# -------------------- FILE UPLOADER --------------------

uploaded_files = st.file_uploader(
    "Upload Resume",
    type=['docx','pdf'],
    accept_multiple_files=True
)

filename = []
predicted = []
skills = []

# -------------------- MAIN PREDICTION LOOP --------------------

for file in uploaded_files:

    text = extract_text(file)
    cleaned = preprocess(text)

    prediction = model.predict(Vectorizer.transform([cleaned]))[0]

    filename.append(file.name)
    predicted.append(prediction)
    skills.append(extract_skills(text))

# -------------------- DISPLAY RESULT --------------------

if len(predicted) > 0:

    result = pd.DataFrame({
        'Uploaded File': filename,
        'Predicted Profile': predicted,
        'Skills': skills
    })

    st.subheader("Prediction Results")
    st.table(result)

# -------------------- FILTER --------------------

    st.subheader("Filter by Profile")
    option = st.selectbox('Select Category', result['Predicted Profile'].unique())

    st.table(result[result['Predicted Profile'] == option])

# -------------------- BAR GRAPH --------------------

    st.subheader("📊 Resume Category Distribution")

    category_counts = pd.Series(predicted).value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)

    ax.set_xlabel("Predicted Categories")
    ax.set_ylabel("Number of Resumes")
    ax.set_title("Resume Count per Predicted Profile")
    plt.xticks(rotation=45)

    st.pyplot(fig)
