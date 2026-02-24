import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Do not truncate pandas output columns
pd.set_option('display.max_columns', None)  

# Load data
fake_data = pd.read_csv("Fake.csv",  encoding="latin1")
true_data = pd.read_csv("True.csv", encoding="latin1")

# print(fake_data.head())

# Add target lables to data
false_label = pd.Series([0] * fake_data.shape[0])
true_label = pd.Series([1] * true_data.shape[0])

fake_data = fake_data.assign(target=false_label)
true_data = true_data.assign(target=true_label)

# print(fake_data.head())
# print(true_data.head())

# Concatenate data into single dataset
data_frames = [fake_data, true_data]
data = pd.concat(data_frames)

target = data["target"]
data = data.drop("target", axis=1)

print(target)
print(data)

# Train/Test Split with 30% test data
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Create CountVectorizer Object
vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20
)

# Pass data through pre-processing pipeline
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train on MultinomialNB
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make Predictions and Evaluate
y_pred = model.predict(X_test_vec)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            xticklabels=['True News','Fake News'],
            yticklabels=['True News','Fake News'])
plt.ylabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17, pad=20)
plt.gca().xaxis.set_label_position('top') 
plt.xlabel('Prediction', fontsize=13)
plt.gca().xaxis.tick_top()

plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()


