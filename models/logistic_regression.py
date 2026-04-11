import numpy as np


def sigmoid(values):
    """Regner ut sigmoid-funksjonen: 1 / (1 + e^(-x))."""
    return 1 / (1 + np.exp(-values))


def predict(x_data, coefficients):
    """Predikerer sannsynligheter med logistisk regresjon.

    Argumenter:
    - x_data      : [n, p]-array med inputdata, n datapunkter og p trekk.
    - coefficients: [p+1]-array der indeks 0 er konstantleddet og resten er vektene.

    Returnerer:
    - [n]-array med predikerte sannsynligheter mellom 0 og 1.
    """
    bias = coefficients[0]
    weights = coefficients[1:]
    return sigmoid(x_data @ weights + bias)
