from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from data.loaders import prepare_data, print_split_stats
from metrics.visualization import scatter_plot
from models.classifier import train_and_evaluate, evaluate_full


def main():
    dataset = load_dataset("ltg/norec")

    train_data, train_labels = prepare_data(dataset, "train")
    dev_data, dev_labels = prepare_data(dataset, "validation")
    test_data, test_labels = prepare_data(dataset, "test")

    print_split_stats("train", train_labels)
    print_split_stats("validation", dev_labels)
    print_split_stats("test", test_labels)

    # Vectorization
    vectorizer = CountVectorizer(max_features=5000)
    vectorizer.fit(train_data)

    train_vectors = vectorizer.transform(train_data)
    dev_vectors = vectorizer.transform(dev_data)
    test_vectors = vectorizer.transform(test_data)

    scatter_plot(train_vectors, train_labels)

    # TF-IDF weighting
    tfidf = TfidfTransformer()
    tfidf.fit(train_vectors)

    train_tfidf = tfidf.transform(train_vectors)
    dev_tfidf = tfidf.transform(dev_vectors)
    test_tfidf = tfidf.transform(test_vectors)

    scatter_plot(train_tfidf, train_labels)

    # Task 3.1 - k-NN with k=1 and k=5000
    for k in [1, 5000]:
        acc, f1 = train_and_evaluate(train_tfidf, train_labels, dev_tfidf, dev_labels, k)
        print(f"k={k}: accuracy={acc:.3f}, f1={f1:.3f}")

    # Task 3.2 - Hyperparameter tuning k=1..20
    print("\nk  | no TF-IDF acc | no TF-IDF f1 | TF-IDF acc | TF-IDF f1")
    print("-" * 65)
    for k in range(1, 21):
        acc, f1 = train_and_evaluate(train_vectors, train_labels, dev_vectors, dev_labels, k)
        acc_tfidf, f1_tfidf = train_and_evaluate(train_tfidf, train_labels, dev_tfidf, dev_labels, k)
        print(f"{k:<3}| {acc:.3f}         | {f1:.3f}        | {acc_tfidf:.3f}      | {f1_tfidf:.3f}")

    # Task 3.3 - Final evaluation with best model (k=20, TF-IDF)
    print("\nFinal evaluation with k=20, TF-IDF:")
    print(f"{'':10} {'accuracy':>10} {'f1':>10} {'precision':>10} {'recall':>10}")
    for name, vectors, labels in [
        ("validation", dev_tfidf, dev_labels),
        ("test", test_tfidf, test_labels),
    ]:
        metrics = evaluate_full(train_tfidf, train_labels, vectors, labels, k=20)
        print(f"{name:10} {metrics['accuracy']:>10.3f} {metrics['f1']:>10.3f} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f}")


if __name__ == "__main__":
    main()
