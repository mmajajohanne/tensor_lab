from data.loaders import load_dataset
from data.preprocessing import (
    tokenize_dataset,
    remove_stopwords,
    get_top_n_words,
    build_vocab,
)
from models.vectorizer import build_cooccurrence_matrix
from metrics.similarity import cosine_similarity
from constants import WORDS, WORDS_SUBSET


def main():
    # 1. last inn og tokeniser datasettet
    dataset = load_dataset("data/NAK_dataset.txt")
    tokenized_dataset = tokenize_dataset(dataset)

    all_tokens = [word for sentence in tokenized_dataset for word in sentence]
    print(f"Antall tokens: {len(all_tokens)}")
    print(f"Antall ordtyper: {len(set(all_tokens))}")

    # 2. fjern stoppord
    filtered_dataset = remove_stopwords(tokenized_dataset)
    filtered_tokens = [word for sentence in filtered_dataset for word in sentence]
    print(f"Antall tokens etter fjerning av stoppord: {len(filtered_tokens)}")
    print(f"Antall ordtyper etter fjerning av stoppord: {len(set(filtered_tokens))}")

    # 3. behold topp 10 000 ord og bygg vokabular
    top_words = get_top_n_words(filtered_dataset, n=10000)
    word_to_int, int_to_word = build_vocab(top_words)

    # 4. bygg sam-forekomstmatrisen
    matrix = build_cooccurrence_matrix(filtered_dataset, word_to_int)
    print(f"Dimensjonene til matrisen: {matrix.shape}")

    # 5. finn de 5 mest like ordene for hvert ord i WORDS_SUBSET
    for word in WORDS_SUBSET:
        if word not in word_to_int:
            print(f"'{word}' er ikke i vokabularet\n")
            continue

        similarities = []
        for other_word in WORDS:
            if other_word in word_to_int and other_word != word:
                sim = cosine_similarity(
                    matrix[word_to_int[word]],
                    matrix[word_to_int[other_word]],
                )
                similarities.append((other_word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"De 5 mest like ordene til '{word}':")
        for i in range(5):
            print(f"  {similarities[i][0]}: {similarities[i][1]:.4f}")
        print()


if __name__ == "__main__":
    main()
