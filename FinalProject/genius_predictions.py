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

TOKEN = "1Y4W8rda4lIgTZL8djnafAnIrZqH-YsiyxVRmdx7Cii1m_h5XO-uWzwjxhnRZZVl"
GENIUS = lyricsgenius.Genius(TOKEN)
MODEL_PATH = "models"
1
# artist = genius.search_artist("Gorillaz", max_songs=3, sort="title")
# print(artist.songs)

# song = genius.search_song(title="Dumb Litty")

# print(song)

# for k,v in song.items():
#     print(k,v)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

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

def predict_new_song(song):
    # Load pre-trained models and vectorizer
    models, vecs = load_models()

    clean_lyrics = preprocess(song.lyrics)

    # print(dir(song))
    print(song.full_title)

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
    while True:
        print("Select an option")
        print("0. Exit Program")
        print("1. Predict new song")
        res = input()

        if res == "0":
            break

        if res == "1":
            print("Please enter the title of a song: ")
            song_name = str(input())
            song_name = song_name.strip()
            song = GENIUS.search_song(title=song_name)
            print("Founds Song: ", song)
            if song is not None:
                predict_new_song(song)
            else:
                print("Error: Song not found")
        else:
            print("Error: Invalid Input")
        

if __name__ == "__main__":
    main()
    