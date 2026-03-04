import nltk
import streamlit as st
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('punkt')

st.title("📚 AI Study Notes Summarizer")

text = st.text_area("Paste your study notes here")

def summarize_text(text, num_sentences=3):
    
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

    nx_graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    summary = " ".join([s for _, s in ranked_sentences[:num_sentences]])

    return summary

if st.button("Summarize"):

    if text.strip() == "":
        st.warning("Please enter notes")
    else:
        summary = summarize_text(text)

        st.subheader("Summary")
        st.write(summary)