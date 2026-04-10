def load_dataset(dataset_path):
    """Åpner tekstfilen og returnerer en liste med linjene i filen."""
    with open(dataset_path, "r") as f:
        dataset = []
        for line in f:
            dataset.append(line.strip())
    return dataset
