import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []
    
    # Make a prediction using a trained model
    def predict(self, X):
        # Compute weighted sum + bias
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply activation function
        return np.where(linear_output >= 0, 1, 0)
    
    # Fit the model to the data
    def fit(self, X, y):
        # Get rows and columns of data
        n_samples, n_features = X.shape
        # Set weight for each feature to 0
        self.weights = np.zeros(n_features)
        # Set bias to 0
        self.bias = 0.0
        # Loop through epochs
        for _ in range(self.epochs):
            errors = 0
            # Loop through each input 
            for xi, target in zip(X, y):
                # Calculate weighted sum + bias
                linear_output = np.dot(xi, self.weights) + self.bias
                # Apply activation function
                y_pred = 1 if linear_output >= 0 else 0
                # Compute error term
                update = self.lr * (target - y_pred)
                # Update weights and bias with error term
                self.weights += update * xi
                self.bias += update
                # track incorrect predictions
                errors += int(update != 0)
            # track total incorrect predictions per epoch
            self.errors_per_epoch.append(errors)
            
def main():
    # Load data
    data = pd.read_csv("spam.csv",  encoding="latin1")
    
    data["target"] = data["v1"].apply(lambda x: 1 if x == "spam" else 0)

    # print(data.head())

    # Split data into train feature and target label
    target = data["target"]
    train = data["v2"]

    # print(target)
    # print(train)

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)
    
    # Create CountVectorizer Object
    n_gram = (2, 2)
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=n_gram,
        max_features=5000
    )
    
    # Pass data through pre-processing pipeline and convert from sparse matrix to dense array for downstream processing
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    
    model = Perceptron()
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)

    # Generate confusion matrix plot
    sns.heatmap(
                cm, annot=True, fmt='g',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam']
                )

    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top') 
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()

    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, 'Confusion Matrix with ngram_range: ' + str(n_gram), ha='center', fontsize=13)
    plt.show()

if __name__ == "__main__":
    main()