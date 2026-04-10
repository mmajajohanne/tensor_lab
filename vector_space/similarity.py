def cosine_similarity(vec1, vec2):
    """Regner ut cosinuslikhet mellom to vektorer uten numpy."""
    dot_product = 0
    for i in range(len(vec1)):
        dot_product += vec1[i] * vec2[i]

    magnitude_vec1 = 0
    magnitude_vec2 = 0
    for i in range(len(vec1)):
        magnitude_vec1 += vec1[i] ** 2
        magnitude_vec2 += vec2[i] ** 2

    magnitude_vec1 = magnitude_vec1**0.5
    magnitude_vec2 = magnitude_vec2**0.5

    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0
    return dot_product / (magnitude_vec1 * magnitude_vec2)
