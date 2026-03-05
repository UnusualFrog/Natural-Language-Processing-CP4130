import re
import numpy as np
import pandas as pd
import os
import joblib
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

# Global random seed
RANDOM_SEED = 42
MODEL_PATH = "models"

# Download NLTK libraries
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Save models and vectorizers
def save_models(models, vectorizers):
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    for name, model in models.items():
        joblib.dump(model, f"{MODEL_PATH}/{name}.joblib")
    
    for name, vectorizer in vectorizers.items():
        joblib.dump(vectorizer, f"{MODEL_PATH}/{name}.joblib")
    
    print(f"Models saved to {MODEL_PATH}/")

# Load models and vectorizers
def load_models():
    models = {
        "lr": joblib.load(f"{MODEL_PATH}/lr.joblib"),
        "nb": joblib.load(f"{MODEL_PATH}/nb.joblib"),
        "svm": joblib.load(f"{MODEL_PATH}/svm.joblib"),
        "rf": joblib.load(f"{MODEL_PATH}/rf.joblib")
    }
    vectorizers = {
        "tfidf_title": joblib.load(f"{MODEL_PATH}/tfidf_title.joblib"),
        "tfidf_lyrics": joblib.load(f"{MODEL_PATH}/tfidf_lyrics.joblib"),
        "tfidf_char": joblib.load(f"{MODEL_PATH}/tfidf_char.joblib")
    }
    return models, vectorizers

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def train_eval():
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv("spotify_songs.csv", encoding="latin1")

    rows_non_eng = df[df["language"] != "en"].index
    df.drop(rows_non_eng, inplace=True)

    rows_latin = df[df["playlist_genre"] == "latin"].index
    df.drop(rows_latin, inplace=True)

    df.dropna(subset="lyrics", inplace=True)

    df_features = df[["track_name", "lyrics"]]
    df_target = df["playlist_genre"]

    # Check class balance
    # class_counts = df["playlist_genre"].value_counts()
    # for k, v in class_counts.items():
    #     print(k, round((v / df.shape[0]) * 100, 2))

    # 70%/30% train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_target,
        random_state=RANDOM_SEED,
        test_size=0.3,
        stratify=df_target
    )

    X_train_clean = X_train.copy()
    X_train_clean["lyrics"] = [preprocess(lyric) for lyric in X_train["lyrics"]]

    X_test_clean = X_test.copy()
    X_test_clean["lyrics"] = [preprocess(lyric) for lyric in X_test["lyrics"]]

    # TF-IDF
    tfidf_title = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2),
        min_df=3, max_df=0.9, sublinear_tf=True
    )
    tfidf_lyrics = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 3),
        min_df=3, max_df=0.85, sublinear_tf=True, max_features=100000
    )
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=3, max_df=0.9, sublinear_tf=True, max_features=50000
    )

    X_train_title = tfidf_title.fit_transform(X_train_clean["track_name"])
    X_train_lyrics = tfidf_lyrics.fit_transform(X_train_clean["lyrics"])
    X_train_char = tfidf_char.fit_transform(X_train_clean["lyrics"])
    X_train_combined = hstack([X_train_title, X_train_lyrics, X_train_char])

    X_test_title = tfidf_title.transform(X_test_clean["track_name"])
    X_test_lyrics = tfidf_lyrics.transform(X_test_clean["lyrics"])
    X_test_char = tfidf_char.transform(X_test_clean["lyrics"])
    X_test_combined = hstack([X_test_title, X_test_lyrics, X_test_char])

    # Model Training
    lr = LogisticRegression(
        max_iter=10000, solver="saga", class_weight="balanced",
        C=1, l1_ratio=0.0, random_state=RANDOM_SEED
    )
    print("========== Training Logistic Regression ==========")
    lr.fit(X_train_combined, y_train)
    lr_pred = lr.predict(X_test_combined)

    class_counts = df_target.value_counts().sort_index()
    class_prior = (1 / class_counts) / (1 / class_counts).sum()

    nb = MultinomialNB(alpha=0.1, class_prior=class_prior.tolist())
    print("========== Training Naive Bayes ==========")
    nb.fit(X_train_combined, y_train)
    nb_pred = nb.predict(X_test_combined)

    svm = LinearSVC(
        C=0.1, max_iter=10000, class_weight="balanced",
        random_state=RANDOM_SEED
    )
    print("========== Training SVM ==========")
    svm.fit(X_train_combined, y_train)
    svm_pred = svm.predict(X_test_combined)

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        min_samples_split=2, max_depth=20,
        random_state=RANDOM_SEED
    )
    print("========== Training Random Forest ==========")
    rf.fit(X_train_combined, y_train)
    rf_pred = rf.predict(X_test_combined)

    # Evaluation
    print(classification_report(y_test, lr_pred, labels=lr.classes_))
    print(classification_report(y_test, nb_pred, labels=nb.classes_))
    print(classification_report(y_test, svm_pred, labels=svm.classes_))
    print(classification_report(y_test, rf_pred, labels=rf.classes_))

    # Return models and vectorizers
    return [lr, nb, svm, rf, tfidf_title, tfidf_lyrics, tfidf_char]

def save_models_and_vec(model_vec):
    lr, nb, svm, rf, tfidf_title, tfidf_lyrics, tfidf_char = model_vec

    models = {"lr": lr, "nb": nb, "svm": svm, "rf": rf}
    vectorizers = {
        "tfidf_title": tfidf_title,
        "tfidf_lyrics": tfidf_lyrics,
        "tfidf_char": tfidf_char
    }
    save_models(models, vectorizers)

if __name__ == "__main__":
    models_vecs = train_eval()
    save_models_and_vec(models_vecs)