import numbers

import numpy as np


def _run_loss_function_tests(input_function, message_infix, test_cases, message_on_pass=False):
    for i, (y_data, predictions, expected) in enumerate(test_cases, start=1):
        try:
            result = input_function(y_data, predictions)
            if result is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`). ")
                return
            if not isinstance(result, numbers.Number):
                print(
                    f"Failed: {message_infix}. Expected number as return type, "
                    f"but got value `{result}` with type `{type(result)}`. "
                )
                return
            if not np.isclose(result, expected, atol=1e-6):
                print(
                    f"Failed: {message_infix}. Test number `{i}` with input y_data=`{y_data}`, "
                    f"predictions=`{predictions}`. Expected `{expected}`, got `{result}`. "
                )
                return
        except Exception as e:
            print(f"Failed: {message_infix}. Test number `{i}` got unexpected error: `{e}`. ")
            return

    if message_on_pass:
        n = len(test_cases)
        print(f"Passed: {message_infix}. All [{n}/{n}] tests passed. ")


def test_calculate_mse(input_function, message_on_pass=False):
    message_infix = "`test_calculate_mse`"
    test_cases = [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0),
        (np.array([1, 2, 3]), np.array([1, 2, 4]), 0.3333333333333333),
        (np.array([-1, 0, 1]), np.array([0, 0, 0]), 0.6666666666666666),
        (np.array([1.5, 2.5, 3.5]), np.array([1.0, 2.0, 4.0]), 0.25),
    ]
    _run_loss_function_tests(input_function, message_infix, test_cases, message_on_pass)


def test_calculate_bce(input_function, message_on_pass=False):
    message_infix = "`test_calculate_bce`"
    test_cases = [
        (np.array([1, 0]), np.array([0.9, 0.1]), 0.10536051565782628),
        (np.array([1, 0]), np.array([0.8, 0.2]), 0.2231435513142097),
        (np.array([0, 1]), np.array([0.3, 0.7]), 0.35667494393873245),
        (np.array([1, 1]), np.array([0.9, 0.9]), 0.10536051565782628),
        (np.array([0, 0]), np.array([0.1, 0.1]), 0.10536051565782628),
    ]
    _run_loss_function_tests(input_function, message_infix, test_cases, message_on_pass)


def test_calculate_accuracy(input_function, message_on_pass=False):
    message_infix = "`test_calculate_accuracy`"
    test_cases = [
        (np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), 1.0),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), 0.0),
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), 0.5),
        (np.array([0, 1, 1, 0, 1]), np.array([0, 1, 0, 0, 1]), 0.8),
        (np.array([1, 0, 1, 0]), np.array([1, 0, 1, 1]), 0.75),
        (np.array([1]), np.array([1]), 1.0),
    ]
    _run_loss_function_tests(input_function, message_infix, test_cases, message_on_pass)


def _run_predict_tests(input_function, message_infix, test_cases, message_on_pass=False):
    for i, (x_data, weights, bias, expected) in enumerate(test_cases, start=1):
        try:
            coefficients = np.concatenate(([bias], weights))
            predictions = input_function(x_data, coefficients)

            if predictions is None:
                print(f"Failed: {message_infix}. Not implemented (`predict()` returned `None`). ")
                return
            if not isinstance(predictions, np.ndarray):
                print(f"Failed: {message_infix}. Expected `np.ndarray` from `predict()`, got `{type(predictions)}`. ")
                return
            if predictions.shape != expected.shape:
                print(
                    f"Failed: {message_infix}. Wrong output shape from `predict()`. "
                    f"Expected shape `{expected.shape}`, got `{predictions.shape}`. "
                )
                return
            if not np.allclose(predictions, expected, atol=1e-6):
                print(
                    f"Failed: {message_infix}. Test number `{i}` with input x_data=`{x_data}`, "
                    f"weights=`{weights}`, bias=`{bias}`. Expected `{expected}`, got `{predictions}`. "
                )
                return

        except Exception as e:
            print(f"Failed: {message_infix}. Test number `{i}` got unexpected error: `{e}`. ")
            return

    if message_on_pass:
        n = len(test_cases)
        print(f"Passed: {message_infix}. All [{n}/{n}] tests passed. ")


def test_predict_linear_regression(input_function, message_on_pass=False):
    message_infix = "`test_predict_linear_regression`"
    test_cases = [
        (
            np.array([[1.0], [2.0], [3.0]]),
            np.array([2.0]),
            1.0,
            np.array([3.0, 5.0, 7.0]),
        ),
        (
            np.array([[1.0, -2.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 2.0, -3.0]]),
            np.array([1.0, -0.5, 2.0]),
            -1.0,
            np.array([1.0, -1.0, -9.0]),
        ),
        (
            np.array(
                [
                    [1.0, 0.0, -1.0, 2.0, 0.5],
                    [0.0, -2.0, 1.0, 0.0, 3.0],
                    [-1.0, 1.0, 0.0, -2.0, 1.0],
                ]
            ),
            np.array([0.5, -1.0, 2.0, -0.5, 1.5]),
            0.0,
            np.array([-1.75, 8.5, 1.0]),
        ),
    ]
    _run_predict_tests(input_function, message_infix, test_cases, message_on_pass)


def test_predict_logistic_regression(input_function, message_on_pass=False):
    message_infix = "`test_predict_logistic_regression`"
    test_cases = [
        (
            np.array([[0.0], [1.0], [2.0]]),
            np.array([1.0]),
            0.0,
            np.array([0.5, 0.73105858, 0.88079708]),
        ),
        (
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            np.array([2.0, -1.0]),
            0.5,
            np.array([0.92414182, 0.37754067, 0.81757448]),
        ),
        (
            np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 2.0, -1.0]]),
            np.array([1.0, -0.5, 2.0]),
            -0.5,
            np.array([0.73105858, 0.37754067, 0.01098694]),
        ),
    ]
    _run_predict_tests(input_function, message_infix, test_cases, message_on_pass)


def test_sigmoid(input_function, message_on_pass=False):
    message_infix = "`test_sigmoid`"
    test_cases = [
        (np.array([0.0]), np.array([0.5])),
        (np.array([1.0]), np.array([0.73105858])),
        (np.array([-1.0]), np.array([np.float64(0.2689414213699951)])),
        (np.array([0.0, 2.0, -2.0]), np.array([0.5, 0.88079708, 0.11920292])),
        (np.array([-10.0, 0.0, 10.0]), np.array([4.53978687e-05, 5.00000000e-01, 9.99954602e-01])),
    ]

    for i, (input_array, expected) in enumerate(test_cases, start=1):
        try:
            result = input_function(input_array)
            if result is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`). ")
                return
            if not isinstance(result, np.ndarray):
                print(f"Failed: {message_infix}. Expected `np.ndarray` as return type, got `{type(result)}`. ")
                return
            if result.shape != expected.shape:
                print(
                    f"Failed: {message_infix}. Wrong output shape for test number `{i}`. "
                    f"Expected shape `{expected.shape}`, got `{result.shape}`. "
                )
                return
            if not np.allclose(result, expected, atol=1e-6):
                print(
                    f"Failed: {message_infix}. Test number `{i}` with input `{input_array}`. "
                    f"Expected `{expected}`, got `{result}`. "
                )
                return

        except Exception as e:
            print(f"Failed: {message_infix}. Test number `{i}` got unexpected error: `{e}`. ")
            return

    if message_on_pass:
        n = len(test_cases)
        print(f"Passed: {message_infix}. All [{n}/{n}] tests passed. ")
