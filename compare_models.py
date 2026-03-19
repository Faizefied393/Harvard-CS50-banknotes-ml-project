import csv

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def load_data(filename):
    evidence = []
    labels = []

    with open(filename, newline="") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            evidence.append([float(value) for value in row[:4]])
            labels.append("Authentic" if row[4] == "0" else "Counterfeit")

    return evidence, labels


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    correct = sum(1 for actual, pred in zip(y_test, predictions) if actual == pred)
    incorrect = len(y_test) - correct
    accuracy = correct / len(y_test)

    print(f"\nModel: {type(model).__name__}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


def main():
    evidence, labels = load_data("banknotes.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=0.4, random_state=42
    )

    models = [
        KNeighborsClassifier(n_neighbors=1),
        Perceptron(),
        svm.SVC(),
        GaussianNB()
    ]

    for model in models:
        evaluate_model(model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()