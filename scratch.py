import numpy as np
import pandas as pd
import random
import re
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the data
df = pd.read_csv(r'C:\Users\USER\PycharmProjects\pythonProject3\Combined_Jobs_Final.csv.zip')

df = df[['Title', 'Job.Description', 'Company']]
df = df.drop_duplicates(subset=['Title', 'Job.Description', 'Company'], keep='first')
df = df.sample(n=1000, random_state=42)
print(df.shape)

# Preprocessing function
ps = PorterStemmer()

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

custom_stopwords = ENGLISH_STOP_WORDS

def cleaning(txt):
    cleaned_txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
    cleaned_txt = re.sub(r'\s+', ' ', cleaned_txt).strip()
    cleaned_txt = cleaned_txt.lower()
    tokens = cleaned_txt.split()
    stemming = [word for word in tokens if word not in custom_stopwords]
    return " ".join(stemming)

df['Title'] = df['Title'].astype(str).apply(lambda x: cleaning(x))
df['Job.Description'] = df['Job.Description'].astype(str).apply(lambda x: cleaning(x))
df['Company'] = df['Company'].astype(str).apply(lambda x: cleaning(x))
df['new_col'] = df['Title'] + " " + df['Job.Description']

# TF-IDF Vectorizer and Cosine Similarity
tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(df['new_col'])
similarity = cosine_similarity(matrix)

# Save the DataFrame and similarity matrix
pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))