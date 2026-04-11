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

# Trekk-seleksjon
feature_sets = [
    ["weight", "acceleration"],
    ["weight", "acceleration", "horsepower", "model_year"],
    ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"],
]

print("\nTrekk-seleksjon:")
for features in feature_sets:
    data = get_auto_mpg_data(columns_to_include=features, perform_scaling=True)
    m = LinearRegression()
    m.fit(data["x_train"], data["y_train"])
    preds = m.predict(data["x_val"])
    mse = calculate_mse(data["y_val"], preds)
    print(f"  {', '.join(features)}"
          f"\n    MSE = {mse:.4f}")

# Logistisk regresjon - Spambase
spam_data = get_spambase_data(
    columns_to_include=["word_freq_free", "char_freq_%24", "capital_run_length_total"],
    perform_scaling=True,
)
x_train = spam_data["x_train"]
y_train = spam_data["y_train"]
x_val = spam_data["x_val"]
y_val = spam_data["y_val"]

spam_model = LogisticRegression()
spam_model.fit(x_train, y_train)

probabilities = spam_model.predict_proba(x_val)[:, 1]
classifications = spam_model.predict(x_val)

bce = calculate_bce(y_val, probabilities)
accuracy = calculate_accuracy(y_val, classifications)
print(f"\nLogistisk regresjon - 3 trekk:")
print(f"  BCE      = {bce:.4f}")
print(f"  Nøyaktighet = {accuracy:.4f}")

# Med alle trekk
spam_data_full = get_spambase_data(columns_to_include=None, perform_scaling=True)
spam_model_full = LogisticRegression()
spam_model_full.fit(spam_data_full["x_train"], spam_data_full["y_train"])

probabilities_full = spam_model_full.predict_proba(spam_data_full["x_val"])[:, 1]
classifications_full = spam_model_full.predict(spam_data_full["x_val"])

bce_full = calculate_bce(spam_data_full["y_val"], probabilities_full)
accuracy_full = calculate_accuracy(spam_data_full["y_val"], classifications_full)
print(f"\nLogistisk regresjon - alle trekk:")
print(f"  BCE      = {bce_full:.4f}")
print(f"  Nøyaktighet = {accuracy_full:.4f}")
