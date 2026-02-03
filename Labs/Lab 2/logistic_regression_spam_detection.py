import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def classify_message(model, vectorizer, message):
    message_vect = vectorizer.transform([message])
    prediction = model.predict(message_vect)
    return "spam" if prediction[0] == 0 else "ham"

def main():
    print("==========Welcome To NLP!============")
    
    # Load data using latin1 encoding to handle non-ASCII characters
    data = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Rename generic column names
    data.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    
    # Map labels from text to numeric values
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    print(data)
    
    # Convert text data into a numeric format
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])
    
    # Seperate target feature from training features
    y = data['label']

    # Split data into training and test sets with a 75/25 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train Model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    print("Accuracy_score" ,accuracy_score(y_test, y_pred))
     
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"[[{cm[0,0]} {cm[0,1]}]")
    print(f" [{cm[1,0]} {cm[1,1]}]]")
    
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    
    message = "Congratulations! You've won a free ticket to Bahamas!"
    print(classify_message(model, vectorizer, message))

if __name__ == "__main__":
    main()