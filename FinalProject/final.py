import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# Do not truncate display of head
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Download NLTK libraries
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # clean text
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # tokenize
    tokens = text.split()
    # stop word removal
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

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
    rows_non_eng = df[df["language"] != "en"].index
    df.drop(rows_non_eng, inplace=True)

    # Remove song which are genre "latin" as they contain non-english lyrics
    rows_latin = df[df["playlist_genre"] == "latin"].index
    df.drop(rows_latin, inplace=True)
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
    class_counts = df["playlist_genre"].value_counts()
    for k,v in class_counts.items():
        print(k, round((v/df.shape[0])*100, 2))

    # 70%/30% train/test split with stratification to account for moderate class imbalance 
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, random_state=42, test_size=0.3, stratify=df_target)

    # print(X_train.head())
    # print(X_test.head())

    # Clean data to remove convert to lowercase, remove numbers, remove punctuation and remove special characters
    X_train_clean = X_train.copy()
    X_train_clean["lyrics"] = [preprocess(lyric) for lyric in X_train["lyrics"]]

    X_test_clean = X_test.copy()
    X_test_clean["lyrics"] = [preprocess(lyric) for lyric in X_test["lyrics"]]

    # print(X_train_clean.head())
    # print(X_test_clean.head())

    #TF-IDF
    tfidf_title = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )

    tfidf_lyrics = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,3),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
        max_features=100000
    )

    X_train_title = tfidf_title.fit_transform(X_train_clean["track_name"])
    X_train_lyrics = tfidf_lyrics.fit_transform(X_train_clean["lyrics"])
    X_train_combined = hstack([X_train_title, X_train_lyrics])

    X_test_title = tfidf_title.transform(X_test_clean["track_name"])
    X_test_lyrics = tfidf_lyrics.transform(X_test_clean["lyrics"])
    X_test_combined = hstack([X_test_title, X_test_lyrics])

    # Model Training
    lr = LogisticRegression(max_iter=10000, solver="saga", class_weight="balanced")
    print("========== Training Logistic Regression ==========")
    lr.fit(X_train_combined, y_train)
    lr_pred = lr.predict(X_test_combined)

    # Calculate class inverse frequency for weighting of NB classes
    class_counts = df_target.value_counts().sort_index()
    class_prior = (1 / class_counts) / (1 / class_counts).sum()

    nb = MultinomialNB(alpha=1.0, class_prior=class_prior.tolist())
    print("========== Training Naive Bayes ==========")
    nb.fit(X_train_combined, y_train)
    nb_pred = nb.predict(X_test_combined)

    svm = LinearSVC( C=0.1, max_iter=10000, class_weight="balanced")
    print("========== Training SVM ==========")
    svm.fit(X_train_combined, y_train)
    svm_pred = svm.predict(X_test_combined)

    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    print("========== Training Random Forest ==========")
    rf.fit(X_train_combined, y_train)
    rf_pred = rf.predict(X_test_combined)

    # Evaluation
    lr_class_report = classification_report(y_test, lr_pred, labels=lr.classes_)
    nb_class_report = classification_report(y_test, nb_pred, labels=nb.classes_)
    svm_class_report = classification_report(y_test, svm_pred, labels=svm.classes_)
    rf_class_report = classification_report(y_test, rf_pred, labels=rf.classes_)

    print(lr_class_report)
    print(nb_class_report)
    print(svm_class_report)
    print(rf_class_report)

if __name__ == "__main__":
    main()

