from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train_and_evaluate(train_vectors, train_labels, dev_vectors, dev_labels, k):
    """Trener en k-NN klassifikator og evaluerer på valideringssettet.

    Argumenter:
    - train_vectors : Vektoriserte treningsdata.
    - train_labels  : Kategorier for treningsdata.
    - dev_vectors   : Vektoriserte valideringsdata.
    - dev_labels    : Kategorier for valideringsdata.
    - k             : Antall naboer i k-NN.

    Returnerer:
    - accuracy : Nøyaktighet på valideringssettet.
    - f1       : F1-mål på valideringssettet.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_vectors, train_labels)
    predictions = knn.predict(dev_vectors)
    accuracy = accuracy_score(dev_labels, predictions)
    f1 = f1_score(dev_labels, predictions, average="macro")
    return accuracy, f1


def evaluate_full(train_vectors, train_labels, eval_vectors, eval_labels, k):
    """Trener en k-NN klassifikator og returnerer accuracy, f1, presisjon og sensitivitet."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_vectors, train_labels)
    predictions = knn.predict(eval_vectors)
    return {
        "accuracy": accuracy_score(eval_labels, predictions),
        "f1": f1_score(eval_labels, predictions, average="macro"),
        "precision": precision_score(eval_labels, predictions, average="macro"),
        "recall": recall_score(eval_labels, predictions, average="macro"),
    }
