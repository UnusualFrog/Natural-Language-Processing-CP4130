from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the dataset
    # Categories to load
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    # Do not include headers, footers or quotes for simplicity
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    # Data preprocessing
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    X_test = vectorizer.transform(newsgroups_test.data)
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    
    # Exploratory Data Analysis
    class_distribution = np.bincount(y_train)
    plt.bar(range(len(class_distribution)), class_distribution)
    plt.xticks(range(len(class_distribution)), newsgroups_train.target_names, rotation=45)
    plt.title('Distribution of Classes in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Documents')
    plt.show()
    
    # Model Training
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Model Evaluation
    y_pred = clf.predict(X_test)
    
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=newsgroups_test.target_names))

if __name__ == "__main__":
    main()