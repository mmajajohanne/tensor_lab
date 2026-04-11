import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from metrics.losses import calculate_accuracy, calculate_bce, calculate_mse
from models.linear_regression import predict as linear_predict
from models.logistic_regression import predict as logistic_predict
from models.logistic_regression import sigmoid
from regression_tests import (
    test_calculate_accuracy,
    test_calculate_bce,
    test_calculate_mse,
    test_predict_linear_regression,
    test_predict_logistic_regression,
    test_sigmoid,
)
from regression_utils import get_auto_mpg_data, get_spambase_data

# Kjor tester
test_calculate_mse(input_function=calculate_mse, message_on_pass=True)
test_calculate_bce(input_function=calculate_bce, message_on_pass=True)
test_calculate_accuracy(input_function=calculate_accuracy, message_on_pass=True)
test_sigmoid(input_function=sigmoid, message_on_pass=True)
test_predict_linear_regression(input_function=linear_predict, message_on_pass=True)
test_predict_logistic_regression(input_function=logistic_predict, message_on_pass=True)

# Lineær regresjon - Auto MPG
mpg_data = get_auto_mpg_data(
    columns_to_include=["weight", "acceleration"],
    perform_scaling=True,
)
x_train = mpg_data["x_train"]
y_train = mpg_data["y_train"]
x_val = mpg_data["x_val"]
y_val = mpg_data["y_val"]

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_val)
mse = calculate_mse(y_val, predictions)
print(f"\nLineær regresjon (weight + acceleration): MSE = {mse:.4f}")
