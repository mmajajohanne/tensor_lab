import numpy as np


def predict(x_data, coefficients):
    """Predikerer verdier med lineær regresjon.

    Argumenter:
    - x_data      : [n, p]-array med inputdata, n datapunkter og p trekk.
    - coefficients: [p+1]-array der indeks 0 er konstantleddet og resten er vektene.

    Returnerer:
    - [n]-array med prediksjoner.
    """
    bias = coefficients[0]
    weights = coefficients[1:]
    return x_data @ weights + bias
