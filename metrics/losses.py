import numpy as np


def calculate_mse(y_data, predictions):
    """Regner ut gjennomsnittlig kvadratfeil (MSE) mellom sanne verdier og prediksjoner."""
    return np.mean((y_data - predictions) ** 2)


def calculate_bce(y_data, predictions):
    """Regner ut binær kryssentropi (BCE) mellom sanne verdier og prediksjoner."""
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(y_data * np.log(predictions) + (1 - y_data) * np.log(1 - predictions))


def calculate_accuracy(y_data, classifications):
    """Regner ut nøyaktighet mellom sanne verdier og klassifikasjoner."""
    return np.mean(y_data == classifications)
