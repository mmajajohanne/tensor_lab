# ---
from helpers import get_norwegian_stopwords
import numpy as np


# åpner tekstfilen og returnerer en liste med linjene i filen
def load_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        dataset = []
        for line in f:
            dataset.append(line.strip())
    return dataset


# splitter en tekst i ord ved å bruke mellomrom som separator
def tokenize(text):
    return text.split()


dataset = load_dataset("NAK_dataset.txt")
tokenized_dataset = []
for text in dataset:
    tokenized_dataset.append(tokenize(text))

all_tokens = []
for sentence in tokenized_dataset:
    for word in sentence:
        all_tokens.append(word)

unique_tokens = set(all_tokens)

print(f"Antall tokens: {len(all_tokens)}")
print(f"Antall ordtyper: {len(unique_tokens)}")
# -----------------------------------------------

# fjerner stoppord
stopwords = set(get_norwegian_stopwords())
filtered_tokens = []
for token in all_tokens:
    if token not in stopwords:
        filtered_tokens.append(token)

unique_filtered_tokens = set(filtered_tokens)

print(f"Antall tokens etter fjerning av stoppord: {len(filtered_tokens)}")
print(f"Antall ordtyper etter fjerning av stoppord: {len(unique_filtered_tokens)}")
# -----------------------------------------------
filtered_dataset = []
for sentence in tokenized_dataset:
    filtered_sentence = []
    for word in sentence:
        if word not in stopwords:
            filtered_sentence.append(word)
    filtered_dataset.append(filtered_sentence)

# fjerner sjeldne ord (beholder bare de 10 000 mest brukte ordene)
word_counts = {}
for sentence in filtered_dataset:
    for word in sentence:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
top_10000 = set(sorted_words[:10000])

# lager to ordbøker: word_to_int og int_to_word for å mappe ord til heltall og omvendt
word_to_int = {}
int_to_word = {}
for i, word in enumerate(sorted_words[:10000]):
    word_to_int[word] = i
    int_to_word[i] = word
# ------------------------------------------------

# matrise som teller hvor mange ganger hvert ord forekommer i hver setning
vocab_size = len(word_to_int)
matrix = np.zeros((vocab_size, vocab_size), dtype=int)
for sentence in filtered_dataset:
    for word_i in sentence:
        if word_i in word_to_int:
            for word_j in sentence:
                if word_j in word_to_int and word_j != word_i:
                    matrix[word_to_int[word_i], word_to_int[word_j]] += 1


# print(f"Dimensjonene til matrisen: {matrix.shape} ")

# henter ut et vanlig ord og finner det mest forekommende ordet som forekommer sammen med det i setningene
# for word in sorted_words[:20]:
#    print(word)

word = "Oslo"
vector = matrix[word_to_int[word]]
most_common_index = np.argmax(vector)
print(f"'{word}' forekommer oftest med '{int_to_word[most_common_index]}'")

word = "kvinne"
vector = matrix[word_to_int[word]]
most_common_index = np.argmax(vector)
print(f"'{word}' forekommer oftest med '{int_to_word[most_common_index]}'")

word = "bil"
vector = matrix[word_to_int[word]]
most_common_index = np.argmax(vector)
print(f"'{word}' forekommer oftest med '{int_to_word[most_common_index]}'")
