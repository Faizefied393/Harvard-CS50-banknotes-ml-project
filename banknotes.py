import csv

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def load_data(filename):
    """
    Load data from a CSV file.
    Returns:
        evidence: list of feature lists
        labels: list of class labels
    """
    evidence = []
    labels = []

    with open(filename, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row

        for row in reader:
            features = [float(value) for value in row[:4]]
            label = "Authentic" if row[4] == "0" else "Counterfeit"

            evidence.append(features)
            labels.append(label)

    return evidence, labels


def train_model(evidence, labels):
    """
    Train and return a k-nearest neighbor classifier.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(actual, predicted):
    """
    Return (correct, incorrect, accuracy)
    """
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    incorrect = sum(1 for a, p in zip(actual, predicted) if a != p)
    accuracy = correct / len(actual)
    return correct, incorrect, accuracy


def main():
    evidence, labels = load_data("banknotes.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        evidence,
        labels,
        test_size=0.4,
        random_state=42
    )

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    correct, incorrect, accuracy = evaluate(y_test, predictions)

    print("Results for model KNeighborsClassifier")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()