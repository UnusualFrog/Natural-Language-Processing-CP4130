import random
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import time
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix

nltk.download('stopwords')

# Load data using latin1 encoding to handle non-ASCII characters
corona_train = pd.read_csv('Corona_NLP_train.csv', encoding='latin-1')
corona_test = pd.read_csv('Corona_NLP_test.csv', encoding='latin-1')

corona_train = corona_train.drop(columns=["UserName", "ScreenName", "Location", "TweetAt"])
corona_test = corona_test.drop(columns=["UserName", "ScreenName", "Location", "TweetAt"])

print(corona_train.head(5))
print(corona_test.head(5))

# Group positive, negative and neutral sentiment
positives_train = corona_train[(corona_train["Sentiment"] == "Positive") | (corona_train["Sentiment"] == "Extremely Positive")]
negatives_train = corona_train[(corona_train["Sentiment"] == "Negative") | (corona_train["Sentiment"] == "Extremely Negative")]
neutral_train = corona_train[(corona_train["Sentiment"] == "Neutral")]

positives_test = corona_test[(corona_test["Sentiment"] == "Positive") | (corona_test["Sentiment"] == "Extremely Positive")]
negatives_test = corona_test[(corona_test["Sentiment"] == "Negative") | (corona_test["Sentiment"] == "Extremely Negative")]
neutral_test = corona_test[(corona_test["Sentiment"] == "Neutral")]

# Encode sentiment values and aggregate positive, negative and neutral sentiment
positives_train["Sentiment"] = 2
neutral_train["Sentiment"] = 1
negatives_train["Sentiment"] = 0

positives_test["Sentiment"] = 2
neutral_test["Sentiment"] = 1
negatives_test["Sentiment"] = 0

print(positives_test, negatives_train)

data = pd.concat([positives_train, positives_test, neutral_train, neutral_test, negatives_train, negatives_test])

data.reset_index(inplace=True)

# Display sentiment of random tweets
# for i in range(0, 10):
#     random_ind = random.randint(0, len(data))
#     print(str(data["OriginalTweet"][random_ind]), end="\nLabel: ")
#     print(str(data["Sentiment"][random_ind]), end="\n\n")

# Calculate distribution
positiveFreqDist = nltk.FreqDist(word for text in data[data["Sentiment"] == 2]["OriginalTweet"] for word in text.lower().split())
neutralFreqDist = nltk.FreqDist(word for text in data[data["Sentiment"] == 1]["OriginalTweet"] for word in text.lower().split())
negativeFreqDist = nltk.FreqDist(word for text in data[data["Sentiment"] == 0]["OriginalTweet"] for word in text.lower().split())

plt.subplots(figsize=(8,6))
plt.title("Most Used Words in Positive Tweets")
positiveFreqDist.plot(50)
# plt.show()

plt.subplots(figsize=(8,6))
plt.title("Most Used Words in Neutral Tweets")
neutralFreqDist.plot(50)
# plt.show()

plt.subplots(figsize=(8,6))
plt.title("Most Used Words in Negative Tweets")
negativeFreqDist.plot(50)
# plt.show()

# Clean data by removing special chars, tokenization, lemmatization, stopword removal and re-joining
cleanedData = []

lemma = WordNetLemmatizer()
swords = stopwords.words("english")
for text in data["OriginalTweet"]:
    
    # Cleaning links
    text = re.sub(r'http\S+', '', text)
    
    # Cleaning everything except alphabetical and numerical characters
    text = re.sub("[^a-zA-Z0-9]"," ",text)
    
    # Tokenizing and lemmatizing
    text = nltk.word_tokenize(text.lower())
    text = [lemma.lemmatize(word) for word in text]
    
    # Removing stopwords
    text = [word for word in text if word not in swords]
    
    # Joining
    text = " ".join(text)
    
    cleanedData.append(text)

for i in range(0,5):
    print(cleanedData[i],end="\n\n")

# Vectorize features
vectorizer = CountVectorizer(max_features=10000)
BOW = vectorizer.fit_transform(cleanedData)

# Train/Test split
x_train,x_test,y_train,y_test = train_test_split(BOW,np.asarray(data["Sentiment"]))

# Train the model
start_time = time.time()

model = SVC()
model.fit(x_train,y_train)

end_time = time.time()
process_time = round(end_time-start_time,2)
print("Fitting SVC took {} seconds".format(process_time))

# Make prediction
predictions = model.predict(x_test)

print("Accuracy of model is {}%".format(accuracy_score(y_test,predictions) * 100))