import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


# download tokenizer
nltk.download('punkt')
# download stopword module
nltk.download('stopwords')

# Load data
data = pd.read_csv("spam.csv",  encoding="latin1")

# print(data.head())

# Split data into train feature and target label
target = data["v1"]
train = data["v2"]

# print(target)
# print(train)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)

# print(X_train)
# print(X_test)

# Tokenize data by splitting on whitespace
X_train_token = X_train.apply(word_tokenize)
X_test_token = X_test.apply(word_tokenize)

# print(X_train_token)
# print(X_test_token)

# Convert each word token to lowercase
X_train_lower = X_train_token.apply(lambda x: [y.lower() for y in x])
X_test_lower = X_test_token.apply(lambda x: [y.lower() for y in x])

# print(X_train_lower)
# print(X_test_lower)

# Remove Punctuation by replacing any non-whitespace or non-word characters with empty strings
X_train_depunc = X_train_lower.apply(lambda x: [
    re.sub(r'[^\w\s]', '', token) for token in x if re.sub(r'[^\w\s]', '', token)
])
X_test_depunc = X_test_lower.apply(lambda x: [
    re.sub(r'[^\w\s]', '', token) for token in x if re.sub(r'[^\w\s]', '', token)
])

# print(X_train_depunc)
# print(X_test_depunc)

# load stopward vocabularys
stop_words = set(stopwords.words('english'))

# Remove any words present in the stopword vocabulary
X_train_stop = X_train_depunc.apply(lambda x: [
    word for word in x if word not in stop_words
])
X_test_stop = X_test_depunc.apply(lambda x: [
    word for word in x if word not in stop_words
])

# print(X_train_stop)
# print(X_test_stop)

# Join tokens back into strings for CountVectorizer processing
X_train_preprocessed = X_train_stop.apply(lambda x: ' '.join(x))
X_test_preprocessed = X_test_stop.apply(lambda x: ' '.join(x))

# vectorizer for converting text to numeric values
# Skip preprocessing as already performed manually
vectorizer = CountVectorizer(
    lowercase=False, tokenizer=lambda x: x.split()
)

# fit vocab to vectorizer
X_train_vectorized = vectorizer.fit_transform(X_train_preprocessed)
# use same features for testing as training
X_test_vectorized = vectorizer.transform(X_test_preprocessed)

# print(X_train_vectorized)
# print(X_test_vectorized)

# train Naive Bayes model
model_nb = MultinomialNB()
model_nb.fit(X_train_vectorized, y_train)

# make prediction
y_pred_nb = model_nb.predict(X_test_vectorized)

# generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_nb)

# show confusion matrix
class_labels_nb = np.unique(y_test)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels_nb, yticklabels=class_labels_nb)
plt.title('Naive Bayes Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Logistic Regression
model_lr = LogisticRegression(max_iter=10000, random_state=42)
model_lr.fit(X_train_vectorized, y_train)

# make prediction
y_pred_lr = model_lr.predict(X_test_vectorized)

# generate confusion matrix
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

# show confusion matrix
class_labels_lr = np.unique(y_test)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Reds', xticklabels=class_labels_lr, yticklabels=class_labels_lr)
plt.title('Logistic Regression Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# SVM
model_svm = SVC(kernel="linear", C=1)
model_svm.fit(X_train_vectorized, y_train)

# make prediction
y_pred_svm = model_svm.predict(X_test_vectorized)

# generate confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# show confusion matrix
class_labels_svm = np.unique(y_test)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels_svm, yticklabels=class_labels_svm)
plt.title('SVM Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()