from helpers import get_norwegian_stopwords


def tokenize(text):
    """Splitter en tekst i ord ved å bruke mellomrom som separator."""
    return text.split()


def tokenize_dataset(dataset):
    """Tokeniserer hele datasettet."""
    tokenized = []
    for text in dataset:
        tokenized.append(tokenize(text))
    return tokenized


def remove_stopwords(tokenized_dataset):
    """Fjerner stoppord fra hver setning i datasettet."""
    stopwords = set(get_norwegian_stopwords())
    filtered = []
    for sentence in tokenized_dataset:
        filtered_sentence = []
        for word in sentence:
            if word not in stopwords:
                filtered_sentence.append(word)
        filtered.append(filtered_sentence)
    return filtered


def get_top_n_words(filtered_dataset, n=10000):
    """Returnerer en sortert liste med de n mest frekvente ordene."""
    word_counts = {}
    for sentence in filtered_dataset:
        for word in sentence:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    return sorted_words[:n]


def build_vocab(top_words):
    """Lager word_to_int og int_to_word ordbøker."""
    word_to_int = {}
    int_to_word = {}
    for i, word in enumerate(top_words):
        word_to_int[word] = i
        int_to_word[i] = word
    return word_to_int, int_to_word
