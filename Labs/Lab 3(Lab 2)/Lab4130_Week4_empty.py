import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import heapq
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
pd.set_option('display.max_colwidth', 120)

"""
Dictionary for encoding string sentiment to numeric values
"""
sentiment_map = {
    "Extremely Negative": -2,
    "Negative": -1,
    "Neutral": 0,
    "Positive": 1,
    "Extremely Positive": 2
}

"""
Function for cleaning data text data to convert to lowercase and remove special characters
"""
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    return text

def load_noBoW():
    # INS. load the dataset without considering the Bag-of-Words model
    df_train = pd.read_csv("Lab4130_Week4_data/Corona_NLP_train.csv",  encoding="latin1")
    df_test = pd.read_csv("Lab4130_Week4_data/Corona_NLP_test.csv",  encoding="latin1")

    # print(df_train.head(5))
    # print(df_train.columns)

    # Drop target and irrelevant feeatures
    df_train_drop = df_train.drop(["UserName", "ScreenName", "Location", "TweetAt", "Sentiment"], axis=1)
    df_test_drop = df_test.drop(["UserName", "ScreenName", "Location", "TweetAt", "Sentiment"], axis=1)

    # print(df_train.head(5))
    # print(df_test.head(5))

    # Convert to lowercase and remove special characters
    df_train_clean = df_train_drop.copy()
    df_train_clean["OriginalTweet"] = df_train_drop["OriginalTweet"].apply(clean_text)

    df_test_clean = df_test_drop.copy()
    df_test_clean["OriginalTweet"] = df_test_drop["OriginalTweet"].apply(clean_text)

    # print(df_train_clean.head(5))
    # print(df_test_clean.head(5))

    # Tokenize
    df_train_token = df_train_clean.copy()
    df_train_token["OriginalTweet"] = df_train_token["OriginalTweet"].apply(word_tokenize)

    df_test_token = df_test_clean.copy()
    df_test_token["OriginalTweet"] = df_test_token["OriginalTweet"].apply(word_tokenize)

    # print(df_train_token.head())
    # print(df_test_token.head())

    # Stopword removal
    stop_words = set(stopwords.words('english'))

    df_train_swr = df_train_token.copy()
    df_train_swr["OriginalTweet"] = [[word for word in row if word not in stop_words] for row in df_train_token["OriginalTweet"]]

    df_test_swr = df_test_token.copy()
    df_test_swr["OriginalTweet"] = [[word for word in row if word not in stop_words] for row in df_test_token["OriginalTweet"]]

    # print(df_train_swr.head())
    # print(df_test_swr.head())

    # Apply Stemming
    stemmer = PorterStemmer()

    df_train_stem = df_train_swr.copy()
    df_train_stem["OriginalTweet"] = [[stemmer.stem(word) for word in row] for row in df_train_stem["OriginalTweet"]]

    df_test_stem = df_test_swr.copy()
    df_test_stem["OriginalTweet"] = [[stemmer.stem(word) for word in row] for row in df_test_stem["OriginalTweet"]]

    # print(df_train_stem.head())
    # print(df_test_stem.head())

    X_train = df_train_stem["OriginalTweet"].apply(lambda row: " ".join(row))
    X_test = df_test_stem["OriginalTweet"].apply(lambda row: " ".join(row))

    # Encode target feature
    y_train_str = df_train["Sentiment"]
    y_train = y_train_str.map(sentiment_map).values

    y_test_str = df_test["Sentiment"]
    y_test = y_test_str.map(sentiment_map).values

    # print(X_train, X_test, y_train, y_test)

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), lowercase=False)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, y_train, X_test, y_test

def load_BoW():
    # Load datasets
    df_train = pd.read_csv("Lab4130_Week4_data/Corona_NLP_train.csv", encoding="latin1")
    df_test = pd.read_csv("Lab4130_Week4_data/Corona_NLP_test.csv", encoding="latin1")
    
    # Drop target and irrelevant features
    df_train_drop = df_train.drop(["UserName", "ScreenName", "Location", "TweetAt", "Sentiment"], axis=1)
    df_test_drop = df_test.drop(["UserName", "ScreenName", "Location", "TweetAt", "Sentiment"], axis=1)
    
    # Clean text
    df_train_clean = df_train_drop.copy()
    df_train_clean["OriginalTweet"] = df_train_drop["OriginalTweet"].apply(clean_text)
    
    df_test_clean = df_test_drop.copy()
    df_test_clean["OriginalTweet"] = df_test_drop["OriginalTweet"].apply(clean_text)

    # print(df_train_clean.head(5))
    # print(df_test_clean.head(5))
    
    # Tokenize
    df_train_token = df_train_clean.copy()
    df_train_token["OriginalTweet"] = df_train_clean["OriginalTweet"].apply(lambda x: nltk.word_tokenize(x))
    
    df_test_token = df_test_clean.copy()
    df_test_token["OriginalTweet"] = df_test_clean["OriginalTweet"].apply(lambda x: nltk.word_tokenize(x))

    # print(df_train_token.head(5))
    # print(df_test_token.head(5))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    df_train_swr = df_train_token.copy()
    df_train_swr["OriginalTweet"] = [[word for word in row if word not in stop_words] 
                                      for row in df_train_token["OriginalTweet"]]
    
    df_test_swr = df_test_token.copy()
    df_test_swr["OriginalTweet"] = [[word for word in row if word not in stop_words] 
                                     for row in df_test_token["OriginalTweet"]]
    
    # print(df_train_swr.head(5))
    # print(df_test_swr.head(5))
    
    # Apply stemming
    stemmer = PorterStemmer()
    
    df_train_stem = df_train_swr.copy()
    df_train_stem["OriginalTweet"] = [[stemmer.stem(word) for word in row] 
                                       for row in df_train_swr["OriginalTweet"]]
    
    df_test_stem = df_test_swr.copy()
    df_test_stem["OriginalTweet"] = [[stemmer.stem(word) for word in row] 
                                      for row in df_test_swr["OriginalTweet"]]
    
    # print(df_train_stem.head(5))
    # print(df_test_stem.head(5))
    
    # Count word frequency
    word_count_train = {}
    
    for row in df_train_stem["OriginalTweet"]:
        for word in row:
            if word not in word_count_train:
                word_count_train[word] = 1
            else:
                word_count_train[word] += 1

    # Get most frequent words
    freq_words_train = heapq.nlargest(5000, word_count_train, key=word_count_train.get)
    
    # Build BoW binary matrix
    X_train = []
    X_test = []
    
    # Count word frequency
    for row in df_train_stem["OriginalTweet"]:
        vector = []
        for word in freq_words_train:
                vector.append(row.count(word))
        X_train.append(vector)
    
    for row in df_test_stem["OriginalTweet"]:
        vector = []
        for word in freq_words_train:
            vector.append(row.count(word))
        X_test.append(vector)
    
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    
    # Encode target feature
    y_train = df_train["Sentiment"].map(sentiment_map).values
    y_test = df_test["Sentiment"].map(sentiment_map).values

    # print(X_train, X_test, y_train, y_test)
    
    return X_train, y_train, X_test, y_test

def SVM(X_train, y_train, X_test, y_test, X_train_bow, y_train_bow, X_test_bow, y_test_bow):
    # INS. Implement Support Vector Machine (SVM) without considering Bag-of-Words model
    # INS. Feel free to use built-in functions
    model_tfidf = LinearSVC(dual='auto')
    model_tfidf.fit(X_train, y_train)
    pred_tdidf = model_tfidf.predict(X_test)
    print(classification_report(y_test, pred_tdidf))

    # INS. Implement Support Vector Machine (SVM) while considering Bag-of-Words model
    # INS. Feel free to use built-in functions
    model_bow = LinearSVC(dual='auto')
    model_bow.fit(X_train_bow, y_train_bow)
    pred_bow = model_bow.predict(X_test_bow)
    print(classification_report(y_test_bow, pred_bow))
    
    return

def LR(X_train, y_train, X_test, y_test, X_train_bow, y_train_bow, X_test_bow, y_test_bow):
    # INS. Implement Logistic Regression (LR) without considering Bag-of-Words model
    # INS. Feel free to use built-in functions
    model_tfidf = LogisticRegression(max_iter=1000)
    model_tfidf.fit(X_train, y_train)
    pred_tdidf = model_tfidf.predict(X_test)
    print(classification_report(y_test, pred_tdidf))

    # INS. Implement Logistic Regression (LR) while considering Bag-of-Words model
    # INS. Feel free to use built-in functions
    model_bow = LogisticRegression(max_iter=500)
    model_bow.fit(X_train_bow, y_train_bow)
    pred_bow = model_bow.predict(X_test_bow)
    print(classification_report(y_test_bow, pred_bow))
    
    return

def main():
    X_train, y_train, X_test, y_test = load_noBoW()
    X_train_bow, y_train_bow, X_test_bow, y_test_bow = load_BoW()

    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)
    # print("==========")
    # print(X_train_bow)
    # print(X_test_bow)
    # print(y_train_bow)
    # print(y_test_bow)

    # Train and Run Models
    print("Running SVM...")
    SVM(X_train, y_train, X_test, y_test, X_train_bow, y_train_bow, X_test_bow, y_test_bow)
    print("Running LR...")
    LR(X_train, y_train, X_test, y_test, X_train_bow, y_train_bow, X_test_bow, y_test_bow)
    

if __name__ == "__main__":
    main()
