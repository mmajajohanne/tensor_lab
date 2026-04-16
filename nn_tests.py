import numbers

import numpy as np


def _run_activation_function_tests(
    input_class, message_infix, test_cases, message_on_pass=False
):
    for i, (x_data, expectedforward, expected_diff) in enumerate(test_cases, start=1):
        try:
            activation = input_class()

            forward = activation(x_data)
            if forward is None:
                print(
                    f"Feilet: {message_infix}. Ikke implementert (returnerte `None`) i forward pass."
                )
                return
            if not isinstance(forward, np.ndarray):
                print(
                    f"Feilet: {message_infix}. Forventet `np.ndarray` fra forward pass, fikk `{type(forward)}`."
                )
                return
            if forward.shape != expectedforward.shape:
                print(
                    f"Feilet: {message_infix}. Feil form i forward pass for test `{i}`. "
                    f"Forventet `{expectedforward.shape}`, fikk `{forward.shape}`."
                )
                return
            if not np.allclose(forward, expectedforward, atol=1e-6):
                print(
                    f"Feilet: {message_infix}. Feil verdier i forward pass for test `{i}`. "
                    f"Forventet `{expectedforward}`, fikk `{forward}`."
                )
                return

            derivative = activation.diff(x_data)
            if derivative is None:
                print(
                    f"Feilet: {message_infix}. Ikke implementert (returnerte `None`) i derivert."
                )
                return
            if not isinstance(derivative, np.ndarray):
                print(
                    f"Feilet: {message_infix}. Forventet `np.ndarray` fra derivert, fikk `{type(derivative)}`."
                )
                return
            if derivative.shape != expected_diff.shape:
                print(
                    f"Feilet: {message_infix}. Feil form i derivert for test `{i}`. "
                    f"Forventet `{expected_diff.shape}`, fikk `{derivative.shape}`."
                )
                return
            if not np.allclose(derivative, expected_diff, atol=1e-6):
                print(
                    f"Feilet: {message_infix}. Feil verdier i derivert for test `{i}`. "
                    f"Forventet `{expected_diff}`, fikk `{derivative}`."
                )
                return

        except Exception as e:
            print(f"Feilet: {message_infix}. Test `{i}` ga uventet feil: `{e}`.")
            return

    if message_on_pass:
        n = len(test_cases)
        print(f"Bestått: {message_infix}. Alle [{n}/{n}] tester bestått.")


def test_sigmoid_class(input_class, message_on_pass=False):
    message_infix = "`test_sigmoid_class`"
    test_cases = [
        (
            np.array([[0.0, 1.0], [-1.0, 2.0]]),
            np.array([[0.5, 0.73105858], [0.26894142, 0.88079708]]),
            np.array([[0.25, 0.19661193], [0.19661193, 0.10499359]]),
        ),
        (
            np.array([[-10.0, 10.0], [0.0, -2.0]]),
            np.array(
                [[4.53978687e-05, 9.99954602e-01], [5.00000000e-01, 1.19202922e-01]]
            ),
            np.array(
                [[4.53958077e-05, 4.53958077e-05], [2.50000000e-01, 1.04993590e-01]]
            ),
        ),
        (
            np.array([[2.0, -2.0], [0.0, 1.0]]),
            np.array([[0.88079708, 0.11920292], [0.5, 0.73105858]]),
            np.array([[0.10499359, 0.10499359], [0.25, 0.19661193]]),
        ),
    ]
    _run_activation_function_tests(
        input_class, message_infix, test_cases, message_on_pass
    )


def test_relu_class(input_class, message_on_pass=False):
    message_infix = "`test_relu_class`"
    test_cases = [
        (
            np.array([[1.0, -1.0], [0.0, 3.0]]),
            np.array([[1.0, 0.0], [0.0, 3.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (
            np.array([[-2.0, -3.0], [4.0, 5.0], [0.0, -1.0]]),
            np.array([[0.0, 0.0], [4.0, 5.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        ),
        (
            np.array([[10.0, -10.0], [-5.0, 2.0]]),
            np.array([[10.0, 0.0], [0.0, 2.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ),
    ]
    _run_activation_function_tests(
        input_class, message_infix, test_cases, message_on_pass
    )


def test_count_parameters(input_class, message_on_pass=False):
    message_infix = "`test_count_parameters`"
    test_cases = [
        ([2, 1], 3),
        ([3, 2], 8),
        ([4, 3, 2], 23),
        ([5, 5, 5], 60),
        ([784, 32, 16, 10], 25818),
        ([10, 10, 10, 10, 10], 440),
    ]
    for i, (layer_sizes, expected) in enumerate(test_cases, start=1):
        try:
            neural_network = input_class(
                layer_sizes=layer_sizes,
                activation_functions=[None] * (len(layer_sizes) - 2),
            )
            result = neural_network.count_parameters()
            if result is None:
                print(f"Feilet: {message_infix}. Ikke implementert (`None` returnert).")
                return
            if not isinstance(result, numbers.Integral):
                print(
                    f"Feilet: {message_infix}. Forventet heltall, fikk `{type(result)}`."
                )
                return
            if result != expected:
                print(
                    f"Feilet: {message_infix}. Test `{i}` med layer_sizes=`{layer_sizes}`. "
                    f"Forventet `{expected}`, fikk `{result}`."
                )
                return
        except Exception as e:
            print(f"Feilet: {message_infix}. Test `{i}` ga uventet feil: `{e}`.")
            return
    if message_on_pass:
        print(
            f"Bestått: {message_infix}. Alle [{len(test_cases)}/{len(test_cases)}] tester bestått."
        )


def test_forward_pass(input_class, relu_class, message_on_pass=False):
    message_infix = "`test_forward_pass`"
    test_cases = []

    layer_sizes = [2, 2, 1]
    activation_functions = [relu_class(), relu_class()]
    x_data = np.array([[1.0, 2.0], [0.0, -1.0]])
    expected_weighted_sums = [
        np.array([[3.0, 3.0], [-1.0, -1.0]]),
        np.array([[6.0], [0.0]]),
    ]
    expected_activations = [
        x_data,
        np.array([[3.0, 3.0], [0.0, 0.0]]),
        np.array([[6.0], [0.0]]),
    ]
    expected_logits = np.array([[6.0], [0.0]])
    test_cases.append(
        (
            layer_sizes,
            x_data,
            expected_weighted_sums,
            expected_activations,
            expected_logits,
        )
    )

    layer_sizes = [3, 2, 2]
    activation_functions = [relu_class(), relu_class()]
    x_data = np.array([[1.0, -1.0, 2.0]])
    expected_weighted_sums = [np.array([[2.0, 2.0]]), np.array([[4.0, 4.0]])]
    expected_activations = [x_data, np.array([[2.0, 2.0]]), np.array([[4.0, 4.0]])]
    expected_logits = np.array([[4.0, 4.0]])
    test_cases.append(
        (
            layer_sizes,
            x_data,
            expected_weighted_sums,
            expected_activations,
            expected_logits,
        )
    )

    for i, (
        layer_sizes,
        x_data,
        expected_weighted_sums,
        expected_activations,
        expected_logits,
    ) in enumerate(test_cases, start=1):
        try:
            neural_network = input_class(
                layer_sizes=layer_sizes,
                activation_functions=activation_functions,
                initialization_method="ones",
            )
            result = neural_network.forward(x_data)

            if result is None:
                print(f"Feilet: {message_infix}. Test `{i}` returnerte None.")
                return
            if not isinstance(result, np.ndarray):
                print(
                    f"Feilet: {message_infix}. Forventet np.ndarray, fikk `{type(result)}`."
                )
                return
            if result.shape != expected_logits.shape:
                print(
                    f"Feilet: {message_infix}. Feil form for test `{i}`. "
                    f"Forventet `{expected_logits.shape}`, fikk `{result.shape}`."
                )
                return
            if not np.allclose(result, expected_logits, atol=1e-6):
                print(
                    f"Feilet: {message_infix}. Feil logit-verdier for test `{i}`. "
                    f"Forventet `{expected_logits}`, fikk `{result}`."
                )
                return
            if not hasattr(neural_network, "weighted_sums") or not hasattr(
                neural_network, "activations"
            ):
                print(
                    f"Feilet: {message_infix}. Mangler `weighted_sums` eller `activations` som attributter."
                )
                return
            if len(neural_network.weighted_sums) != len(expected_weighted_sums):
                print(
                    f"Feilet: {message_infix}. Feil antall weighted_sums. "
                    f"Forventet `{len(expected_weighted_sums)}`, fikk `{len(neural_network.weighted_sums)}`."
                )
                return
            if len(neural_network.activations) != len(expected_activations):
                print(
                    f"Feilet: {message_infix}. Feil antall aktivasjoner. "
                    f"Forventet `{len(expected_activations)}`, fikk `{len(neural_network.activations)}`."
                )
                return
            for j, (z_expected, z_actual) in enumerate(
                zip(expected_weighted_sums, neural_network.weighted_sums)
            ):
                if not np.allclose(z_expected, z_actual, atol=1e-6):
                    print(
                        f"Feilet: {message_infix}. Feil weighted_sum i lag {j + 1}, test `{i}`. "
                        f"Forventet `{z_expected}`, fikk `{z_actual}`."
                    )
                    return
            for j, (a_expected, a_actual) in enumerate(
                zip(expected_activations, neural_network.activations)
            ):
                if not np.allclose(a_expected, a_actual, atol=1e-6):
                    print(
                        f"Feilet: {message_infix}. Feil aktivasjon i lag {j}, test `{i}`. "
                        f"Forventet `{a_expected}`, fikk `{a_actual}`."
                    )
                    return
        except Exception as e:
            print(f"Feilet: {message_infix}. Test `{i}` ga uventet feil: `{e}`.")
            return

    if message_on_pass:
        print(
            f"Bestått: {message_infix}. Alle [{len(test_cases)}/{len(test_cases)}] tester bestått."
        )


def test_predict(input_class, relu_class, message_on_pass=False):
    message_infix = "`test_predict`"
    test_cases = []

    layer_sizes = [2, 2, 2]
    activation_functions = [relu_class(), relu_class()]
    x_data = np.array([[1.0, 2.0], [0.0, -1.0], [2.0, 2.0]])
    expected_predictions = np.array([0, 0, 0])
    test_cases.append((layer_sizes, x_data, expected_predictions))

    layer_sizes = [3, 3, 3]
    activation_functions = [relu_class(), relu_class()]
    x_data = np.array([[1.0, -2.0, 3.0]])
    expected_predictions = np.array([0])
    test_cases.append((layer_sizes, x_data, expected_predictions))

    for i, (layer_sizes, x_data, expected_predictions) in enumerate(
        test_cases, start=1
    ):
        try:
            model = input_class(
                layer_sizes=layer_sizes,
                activation_functions=activation_functions,
                initialization_method="ones",
            )
            result = model.predict(x_data)

            if result is None:
                print(f"Feilet: {message_infix}. Test `{i}` returnerte None.")
                return
            if not isinstance(result, np.ndarray):
                print(
                    f"Feilet: {message_infix}. Forventet np.ndarray, fikk `{type(result)}`."
                )
                return
            if result.ndim != 1:
                print(
                    f"Feilet: {message_infix}. Forventet 1D-array, fikk form `{result.shape}`."
                )
                return
            if result.shape[0] != x_data.shape[0]:
                print(
                    f"Feilet: {message_infix}. Feil antall prediksjoner. "
                    f"Forventet `{x_data.shape[0]}`, fikk `{result.shape[0]}`."
                )
                return
            if not np.array_equal(result, expected_predictions):
                print(
                    f"Feilet: {message_infix}. Feil prediksjoner i test `{i}`. "
                    f"Forventet `{expected_predictions}`, fikk `{result}`."
                )
                return
        except Exception as e:
            print(f"Feilet: {message_infix}. Test `{i}` ga uventet feil: `{e}`.")
            return

    if message_on_pass:
        print(
            f"Bestått: {message_infix}. Alle [{len(test_cases)}/{len(test_cases)}] tester bestått."
        )


def _run_loss_function_tests(
    input_function, message_infix, test_cases, message_on_pass=False
):
    for i, (y_data, predictions, expected) in enumerate(test_cases, start=1):
        try:
            result = input_function(y_data, predictions)
            if result is None:
                print(
                    f"Feilet: {message_infix}. Ikke implementert (returnerte `None`)."
                )
                return
            if not isinstance(result, numbers.Number):
                print(
                    f"Feilet: {message_infix}. Forventet tall som returtype, "
                    f"fikk verdi `{result}` med type `{type(result)}`."
                )
                return
            if not np.isclose(result, expected, atol=1e-6):
                print(
                    f"Feilet: {message_infix}. Test `{i}` med y_data=`{y_data}`, "
                    f"predictions=`{predictions}`. Forventet `{expected}`, fikk `{result}`."
                )
                return
        except Exception as e:
            print(f"Feilet: {message_infix}. Test `{i}` ga uventet feil: `{e}`.")
            return

    if message_on_pass:
        n = len(test_cases)
        print(f"Bestått: {message_infix}. Alle [{n}/{n}] tester bestått.")


def test_calculate_multiclass_cross_entropy_loss(input_function, message_on_pass=False):
    message_infix = "`test_calculate_multiclass_cross_entropy_loss`"
    test_cases = [
        (
            np.array([0, 1]),
            np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
            0.0359762955725428,
        ),
        (
            np.array([0, 2, 2]),
            np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [4.0, 0.0, 0.0]]),
            2.702643041017243,
        ),
        (
            np.array([0, 1, 2]),
            np.zeros((3, 3)),
            1.0986122886681098,
        ),
        (
            np.array([0, 3, 1, 2]),
            np.array(
                [
                    [3.0, 1.0, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.5, 2.0, 0.5, 0.0],
                    [0.0, 0.0, 2.0, 1.0],
                ]
            ),
            0.6438389517485247,
        ),
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
