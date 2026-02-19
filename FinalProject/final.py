import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string
from num2words import num2words
from nltk.tokenize import word_tokenize
import nltk

# Do not truncate display of head
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Download NLTK libraries
nltk.download('punkt_tab')

def clean_text(text):
    # print(text)
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', lambda match: num2words(int(match.group())), text)  # Convert numbers to text representations
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    return text

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
    df_features = df[
        ["track_name", "track_artist", "lyrics", "danceability",
            "energy", "key", "loudness", "mode", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "language"]
        ]
    df_target = df["playlist_genre"]

    # print(df_features.head())
    # print(df_target.head())

    # Check class balance
    class_counts = df["playlist_genre"].value_counts()
    for k,v in class_counts.items():
        print(k, round((v/df.shape[0])*100, 2))

    # 70%/30% train/test split with stratification to account for moderate class imbalance 
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, random_state=42, test_size=0.3, stratify=df_target)

    # print(X_train.head(20))
    # print(X_test.head(20))

    # Clean data to remove convert to lowercase, convert numbers to words, remove punctuation and remove special characters
    X_train_clean = X_train.copy()
    X_train_clean["lyrics"] = [clean_text(lyric) for lyric in X_train["lyrics"]]

    X_test_clean = X_test.copy()
    X_test_clean["lyrics"] = [clean_text(lyric) for lyric in X_test["lyrics"]]

    # print(X_train_clean.head(20))
    # print(X_test_clean.head(20))

    # Tokenize data
    X_train_token = X_train_clean.copy()
    X_test_token = X_test_clean.copy()

    X_train_token["lyrics"] = [word_tokenize(lyric) for lyric in X_train_token["lyrics"]]
    X_test_token["lyrics"] = [word_tokenize(lyric) for lyric in X_test_token["lyrics"]]

    # print(X_train_token.head())
    # print(X_test_token.head())



if __name__ == "__main__":
    main()

