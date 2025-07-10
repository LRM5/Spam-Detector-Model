import requests 
import pandas as pd 
import zipfile
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import io
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

#Get Dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

response = requests.get(url)
if response.status_code == 200:
    print("Download Successful")
else:
    print("Failed to download the dataset")
    
#Extract Dataset
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection") 
    print("Extraction successful")

extracted_files = os.listdir("sms_spam_collection")
print("Extracted files:", extracted_files)

#Load Dataset
df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header = None,
    names =["level", "message"],
)

# Display basic information about dataset
print("-------------------- HEAD ------------------------")
print(df.head())
print("-------------------- DESCRIBE ------------------------")
print(df.describe())
print("-------------------- INFO ------------------------")
print(df.info())

# Check for Duplicates

print("Duplicated entries:", df.duplicated().sum())

df = df.drop_duplicates()


# Data Preprocessing 

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

print("=== BEFORE ANY PREPROCESSING ===")
print(df.head(5))

df["message"] = df["message"].str.lower()
print("\n=== AFTER LOWERCASE ===")
print(df["message"].head(5))

# Remove non-essential punctuation and numbers, keep useful symbols like $ and !

df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
print("\n=== AFTER REMOVING PUNCTUATION & NUMBERS (except $ and !) ===")
print(df["message"].head(5))

# Tokenization (Seperates each word into tokens)
df["message"] = df["message"].apply(word_tokenize)
print("\n=== AFTER TOKENIZATION ===")
print(df["message"].head(5))

# Removes stopwords like (and, until, the) words that dont offer much meaning
stop_words = set(stopwords.words("english"))

df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
print("\n=== AFTER REMOVING STOPWORDS ===")
print(df["message"].head(5))              

# Stemming (Normalizes the words by reducing them to base form i.e running to run)

stemmer = PorterStemmer()
df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
print("\n=== AFTER STEMMING ===")
print(df["message"].head(5))

# Rejoining the tokens back into a single string

df["message"] = df["message"].apply(lambda x: " ".join(x))
print("\n=== AFTER REJOINING TOKENS===")
print(df["message"].head(5))

# Initialize CountVeoctorizer with bigrams, min_df, and max_df to focus on relevant terms
vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))

# Fit and transform the message column
X = vectorizer.fit_transform(df["message"])

# Labels (target variable)
y = df["level"].apply(lambda x: 1 if x == "spam" else 0) # Converts the labels to 1 or 0

pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])

param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,

    scoring="f1"
)

grid_search.fit(df["message"], y)

best_model = grid_search.best_estimator_
print("Best model parameters:", grid_search.best_params_)

# Preprocessing Evaluation Messages
new_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
]

def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

processed_messages = [preprocess_message(msg) for msg in new_message]

X_new = best_model.named_steps["vectorizer"].transform(processed_messages)

predictions = best_model.named_steps["classifier"].predict(X_new)
prediction_probabilities = best_model.named_steps["classifier"].predict_proba(X_new)

for i, msg in enumerate(new_messages):
    prediction = "Spam" if predictions[i] == 1 else "Non-Spam"
    spam_probability = prediction_probabilities[i][1] 
    ham_probability = prediction_probabilities[i][0]

    print(f"Message: {msg}")
    print(f"Prediction: {prediction}")
    print(f"Spam Probability: {spam_probability:.2f}")
    print(f"Not-Spam Probability: {ham_probability:.2f}")
    print("-" * 50)

import joblib

model_filename = "spam_detection_model.joblib"
joblib.dump(best_model, model_filename)

print(f"Model saved to {model_filename}")

import json 
import requests

url =  "URL"

model_file_path = "spam_detection_model path"

with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

    print(json.dumps(response.json(), indent=4))
