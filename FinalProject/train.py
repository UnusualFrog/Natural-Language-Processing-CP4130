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
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt

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

# Performing all pre-processing on text data
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)    # Replace digits with blank space
    text = re.sub(r'[^\w\s]', ' ', text) # Replce anything that is not a word or whitespace with blank space
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words] # Remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens] # Lemmatize words to convert to base form
    return " ".join(tokens) # Reconstruct tokens into string sentence

# Train and evaluate models on training data
def train_eval():
    # Set a random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Load data from csv file
    df = pd.read_csv("spotify_songs.csv", encoding="latin1")

    # Remove non-english languages
    rows_non_eng = df[df["language"] != "en"].index
    df.drop(rows_non_eng, inplace=True)

    # Remove songs from the "latin" genre as they tend to contain a mix of spanish and english
    rows_latin = df[df["playlist_genre"] == "latin"].index
    df.drop(rows_latin, inplace=True)

    # Remove null rows 
    df.dropna(subset="lyrics", inplace=True)

    # Seperate training and target features
    df_features = df[["track_name", "lyrics"]]
    df_target = df["playlist_genre"]

    # Check class balance
    # class_counts = df["playlist_genre"].value_counts()
    # for k, v in class_counts.items():
    #     print(k, round((v / df.shape[0]) * 100, 2))

    # 70%/30% train/test split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_target,
        random_state=RANDOM_SEED,
        test_size=0.3,
        stratify=df_target
    )

    # Pre-process lyrics column for training data
    X_train_clean = X_train.copy()
    X_train_clean["lyrics"] = [preprocess(lyric) for lyric in X_train["lyrics"]]

    # Pre-process lyrics column for testing data
    X_test_clean = X_test.copy()
    X_test_clean["lyrics"] = [preprocess(lyric) for lyric in X_test["lyrics"]]

    # TF-IDF

    # Title vectorizer uses unigrams and bigrams
    tfidf_title = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2),
        min_df=3, max_df=0.9, sublinear_tf=True
    )

    # Lyrics vectorizer uses unigrams, bigrams, and trigrams
    tfidf_lyrics = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 3),
        min_df=3, max_df=0.9, sublinear_tf=True, max_features=100000
    )
    # Lyrics vectorizer on a character-by-character basis using trigrams, quadgrams and quintgrams
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=3, max_df=0.9, sublinear_tf=True, max_features=50000
    )

    # Fit training data to vocabulary and combine all three vocabularies into one
    X_train_title = tfidf_title.fit_transform(X_train_clean["track_name"])
    X_train_lyrics = tfidf_lyrics.fit_transform(X_train_clean["lyrics"])
    X_train_char = tfidf_char.fit_transform(X_train_clean["lyrics"])
    X_train_combined = hstack([X_train_title, X_train_lyrics, X_train_char])

    # Transform testing data to vocabulary and combine all three vocabularies into one
    X_test_title = tfidf_title.transform(X_test_clean["track_name"])
    X_test_lyrics = tfidf_lyrics.transform(X_test_clean["lyrics"])
    X_test_char = tfidf_char.transform(X_test_clean["lyrics"])
    X_test_combined = hstack([X_test_title, X_test_lyrics, X_test_char])

    # Model Training
    lr = LogisticRegression(
        max_iter=10000, 
        solver="saga", 
        class_weight="balanced",
        random_state=RANDOM_SEED
    )
    print("========== Training Logistic Regression ==========")
    lr.fit(X_train_combined, y_train)
    lr_pred = lr.predict(X_test_combined)

    class_counts = y_train.value_counts().sort_index()
    class_prior = (1 / class_counts) / (1 / class_counts).sum()

    nb = MultinomialNB(
        alpha=0.1,
        class_prior=class_prior.tolist()
        )
    
    print("========== Training Naive Bayes ==========")
    nb.fit(X_train_combined, y_train)
    nb_pred = nb.predict(X_test_combined)

    svm = LinearSVC(
        C=0.1,
        max_iter=10000,
        class_weight="balanced",
        random_state=RANDOM_SEED
    )

    print("========== Training SVM ==========")
    svm.fit(X_train_combined, y_train)
    svm_pred = svm.predict(X_test_combined)

    rf = RandomForestClassifier(
        n_estimators=200, 
        class_weight="balanced",
        max_depth=20,
        random_state=RANDOM_SEED
    )

    print("========== Training Random Forest ==========")
    rf.fit(X_train_combined, y_train)
    rf_pred = rf.predict(X_test_combined)

    # Evaluate models 
    print(classification_report(y_test, lr_pred, labels=lr.classes_))
    print(classification_report(y_test, nb_pred, labels=nb.classes_))
    print(classification_report(y_test, svm_pred, labels=svm.classes_))
    print(classification_report(y_test, rf_pred, labels=rf.classes_))

    models = ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"]
    predictions = [lr_pred, nb_pred, svm_pred, rf_pred]

    accuracies = [accuracy_score(y_test, pred) for pred in predictions]
    f1_scores = [f1_score(y_test, pred, average="weighted") for pred in predictions]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10,6))

    plt.bar(x - width/2, accuracies, width, label="Accuracy")
    plt.bar(x + width/2, f1_scores, width, label="F1 Score")

    plt.xticks(x, models, rotation=20)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Model Performance Comparison")

    all_scores = accuracies + f1_scores
    plt.ylim(min(all_scores) - 0.02, max(all_scores) + 0.02)

    for i, v in enumerate(accuracies):
        plt.text(i - width/2, v + 0.001, f"{v:.3f}", ha="center")

    for i, v in enumerate(f1_scores):
        plt.text(i + width/2, v + 0.001, f"{v:.3f}", ha="center")

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Return models and vectorizers 
    return [lr, nb, svm, rf, tfidf_title, tfidf_lyrics, tfidf_char]

# Save models and vocabs to file
def save_models_and_vec(model_vec):
    lr, nb, svm, rf, tfidf_title, tfidf_lyrics, tfidf_char = model_vec

    models = {"lr": lr, "nb": nb, "svm": svm, "rf": rf}
    vectorizers = {
        "tfidf_title": tfidf_title,
        "tfidf_lyrics": tfidf_lyrics,
        "tfidf_char": tfidf_char
    }
    save_models(models, vectorizers)

# Train model, evaluate and save to file
if __name__ == "__main__":
    models_vecs = train_eval()
    save_models_and_vec(models_vecs)