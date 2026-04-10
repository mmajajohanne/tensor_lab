def load_dataset(dataset_path):
    """Åpner tekstfilen og returnerer en liste med linjene i filen."""
    with open(dataset_path, "r") as f:
        dataset = []
        for line in f:
            dataset.append(line.strip())
    return dataset


def prepare_data(dataset, split):
    """Henter ønsket data fra NoReC-datasettet.

    Argumenter:
    - dataset : NoReC-datasettet.
    - split   : En streng som spesifiserer hvilken splitt vi ønsker.

    Returnerer:
    - data   : En liste over dokument-tekstene fra spesifisert splitt.
    - labels : En liste over hvilken kategori dokumentet tilhører.
    """
    categories = {"games", "restaurants", "literature"}
    data = []
    labels = []
    for item in dataset[split]:
        if item["category"] in categories:
            data.append(item["text"])
            labels.append(item["category"])
    return data, labels


def print_split_stats(name, labels):
    """Printer antall dokumenter og kategorifordeling for en splitt."""
    total = len(labels)
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    print(f"{name}: {total} documents")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count} ({count / total:.1%})")
