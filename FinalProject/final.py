import re
import string
import numpy as np
import pandas as pd
from num2words import num2words
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack

# Do not truncate display of head
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Download NLTK libraries
nltk.download('punkt_tab')
nltk.download('stopwords')

def clean_text(text):
    # print(text)
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', lambda match: num2words(int(match.group())), text)  # Convert numbers to text representations
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    return text

def clean_title(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', lambda match: num2words(int(match.group())), text)  # Convert numbers to text representations
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with whitespace
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = " ".join(text.split()) # Normalize whitespace
    return text

def genius_lookup(text):
    pass

def main():
    df = pd.read_csv("spotify_songs.csv",  encoding="latin1")
    # print(df.columns)
    # print(df.head())

    # Print unique genres and languages
    # unique_genre = df["playlist_genre"].unique()
    # print(unique_genre)
    
    # Remove non-english languages
    # print(df["language"].count())
    rows = df[df["language"] != "en"].index
    df.drop(rows, inplace=True)
    # print(df["language"].count())

    # Remove rows with NaN lyrics
    # for col in df:
    #     print(col, df[col].isna().sum())
    df.dropna(subset="lyrics", inplace=True)
    # print("===========")
    # for col in df:
    #     print(col, df[col].isna().sum())
    
    # Select relevant training and target features
    df_features = df[["track_name", "lyrics"]]
    df_target = df["playlist_genre"]

    # print(df_features.head())
    # print(df_target.head())

    # Check class balance
    # class_counts = df["playlist_genre"].value_counts()
    # for k,v in class_counts.items():
    #     print(k, round((v/df.shape[0])*100, 2))

    # 70%/30% train/test split with stratification to account for moderate class imbalance 
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, random_state=42, test_size=0.3, stratify=df_target)

    # print(X_train.head())
    # print(X_test.head())

    # Clean data to remove convert to lowercase, convert numbers to words, remove punctuation and remove special characters
    X_train_clean = X_train.copy()
    X_train_clean["lyrics"] = [clean_text(lyric) for lyric in X_train["lyrics"]]
    X_train_clean["track_name"] = [clean_title(title) for title in X_train["track_name"]]

    X_test_clean = X_test.copy()
    X_test_clean["lyrics"] = [clean_text(lyric) for lyric in X_test["lyrics"]]
    X_test_clean["track_name"] = [clean_title(title) for title in X_test["track_name"]]

    # print(X_train_clean.head())
    # print(X_test_clean.head())

    # Tokenize data
    X_train_token = X_train_clean.copy()
    X_test_token = X_test_clean.copy()

    X_train_token["lyrics"] = [word_tokenize(lyric) for lyric in X_train_token["lyrics"]]
    X_test_token["lyrics"] = [word_tokenize(lyric) for lyric in X_test_token["lyrics"]]

    # print(X_train_token.head())
    # print(X_test_token.head())

    # Optional Stop Word Removal
    stop_words = set(stopwords.words('english'))
    X_train_stop = X_train_token.copy()
    X_test_stop = X_test_token.copy()

    # X_train_stop["lyrics"] = [word for word in X_train_stop["lyrics"] if word not in stop_words]
    # X_test_stop["lyrics"] = [word for word in X_test_stop["lyrics"] if word not in stop_words]
    X_train_stop["lyrics"] = X_train_stop["lyrics"].apply(lambda row: [word for word in row if word not in stop_words])
    X_test_stop["lyrics"] = X_test_stop["lyrics"].apply(lambda row: [word for word in row if word not in stop_words])

    # print(X_train_stop.head())
    # print(X_test_stop.head())

    # Stemming
    porter_stemmer = PorterStemmer()
    X_train_stem = X_train_token.copy()
    X_test_stem = X_test_token.copy()

    X_train_stem["lyrics"] = X_train_stem["lyrics"].apply(lambda row: [porter_stemmer.stem(word) for word in row])
    X_test_stem["lyrics"] = X_test_stem["lyrics"].apply(lambda row: [porter_stemmer.stem(word) for word in row])

    # Re-join tokens for TF-IDF consumption
    X_train_stem["lyrics"] = X_train_stem["lyrics"].apply(" ".join)
    X_test_stem["lyrics"] = X_test_stem["lyrics"].apply(" ".join)

    # print(X_train_stem.head())
    # print(X_test_stem.head())

    #TF-IDF
    # Unigrams with more than 1 occurance
    title_vectorizer = TfidfVectorizer(
        ngram_range=(1,1),
        min_df=2,
        lowercase=False
    )
    # Unigrams and bigrams with more than 4 occurances
    lyrics_vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=5,
        lowercase=False
    )

    X_train_title = title_vectorizer.fit_transform(X_train_stem["track_name"])
    X_train_lyrics = lyrics_vectorizer.fit_transform(X_train_stem["lyrics"])
    X_train_combined = hstack([X_train_title, X_train_lyrics])

    X_test_title = title_vectorizer.transform(X_test_stem["track_name"])
    X_test_lyrics = lyrics_vectorizer.transform(X_test_stem["lyrics"])
    X_test_combined = hstack([X_test_title, X_test_lyrics])

    # Model Training
    lr = LogisticRegression(max_iter=10000, solver="saga", class_weight="balanced")
    lr.fit(X_train_combined, y_train)
    lr_pred = lr.predict(X_test_combined)

    # Calculate class inverse frequency for weighting of NB classes
    class_counts = df_target.value_counts().sort_index()
    class_prior = (1 / class_counts) / (1 / class_counts).sum()

    nb = MultinomialNB(alpha=1.0, class_prior=class_prior.tolist())
    nb.fit(X_train_combined, y_train)
    nb_pred = nb.predict(X_test_combined)

    svm = LinearSVC( C=1, max_iter=10000, class_weight="balanced")
    svm.fit(X_train_combined, y_train)
    svm_pred = svm.predict(X_test_combined)

    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    rf.fit(X_train_combined, y_train)
    rf_pred = rf.predict(X_test_combined)

    # Evaluation
    target_names = df_target.unique().tolist()
    lr_class_report = classification_report(y_test, lr_pred, target_names=target_names)
    nb_class_report = classification_report(y_test, nb_pred, target_names=target_names)
    svm_class_report = classification_report(y_test, svm_pred, target_names=target_names)
    rf_class_report = classification_report(y_test, rf_pred, target_names=target_names)

    print(lr_class_report)
    print(nb_class_report)
    print(svm_class_report)
    print(rf_class_report)




if __name__ == "__main__":
    main()

