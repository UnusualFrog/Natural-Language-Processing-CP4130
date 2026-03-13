import lyricsgenius
import joblib
import re
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK libraries
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Genius API 
TOKEN = "1Y4W8rda4lIgTZL8djnafAnIrZqH-YsiyxVRmdx7Cii1m_h5XO-uWzwjxhnRZZVl"
GENIUS = lyricsgenius.Genius(TOKEN)
MODEL_PATH = "models"

# Apply pre-processing to text data to convert to lowercase, remove special characters and stopwords and lemmatize words
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Load saved models and vocabularies
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

def predict_new_song(song, models, vecs):
    # Preprocess data
    clean_lyrics = preprocess(song.lyrics)

    # print(dir(song))
    print(song.full_title)

    # Transform unseen data to match pre-trained vocabulary
    title_vec = vecs["tfidf_title"].transform([song.full_title])
    lyrics_vec = vecs["tfidf_lyrics"].transform([clean_lyrics])
    char_vec = vecs["tfidf_char"].transform([clean_lyrics])
    combined = hstack([title_vec, lyrics_vec, char_vec])

    for name, model in models.items():
        prediction = model.predict(combined)[0]
        
        # lr, nb, and rf use predict_proba, svm uses decision_function
        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(combined)[0])
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(combined)[0]
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
            confidence = max(scores_norm)
        
        print(f"{name}: {prediction} ({confidence:.2%})")

def main():
    models, vecs = load_models()
    
    # Text UI for predicting new data
    while True:
        print("Select an option")
        print("0. Exit Program")
        print("1. Predict new song")
        res = input()

        # Exit system
        if res == "0":
            break
        
        # Predict new song
        if res == "1":
            print("Please enter the title of a song (and optionally the artist name): ")
            song_name = str(input())
            song_name = song_name.strip()

            # Handle empty input
            if not song_name:
                print("Error: Please enter a song name")
                continue

            song = GENIUS.search_song(title=song_name)
            print("Founds Song: ", song)

            # Predict on new song if found
            if song is not None:
                predict_new_song(song, models, vecs)
            else:
                print("Error: Song not found")
        else:
            print("Error: Invalid Input")
        

if __name__ == "__main__":
    main()
    