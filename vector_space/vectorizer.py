import numpy as np


def build_cooccurrence_matrix(filtered_dataset, word_to_int):
    """Bygger en sam-forekomstmatrise basert på setninger."""
    vocab_size = len(word_to_int)
    matrix = np.zeros((vocab_size, vocab_size), dtype=int)
    for sentence in filtered_dataset:
        for word_i in sentence:
            if word_i in word_to_int:
                for word_j in sentence:
                    if word_j in word_to_int and word_j != word_i:
                        matrix[word_to_int[word_i], word_to_int[word_j]] += 1
    return matrix
