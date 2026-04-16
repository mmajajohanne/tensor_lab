from models.activations import ReLU, Sigmoid
from nn_plotting import plot_mislabeled_mnist_images, plot_random_mnist_images
from nn_tests import (
    test_calculate_accuracy,
    test_calculate_multiclass_cross_entropy_loss,
    test_count_parameters,
    test_forward_pass,
    test_predict,
    test_relu_class,
    test_sigmoid_class,
)
from nn_utils import (
    NeuralNetwork,
    calculate_accuracy,
    calculate_multiclass_cross_entropy,
    load_mnist_data,
)

# Kjør tester
test_sigmoid_class(input_class=Sigmoid, message_on_pass=True)
test_relu_class(input_class=ReLU, message_on_pass=True)
test_count_parameters(input_class=NeuralNetwork, message_on_pass=True)
test_forward_pass(input_class=NeuralNetwork, relu_class=ReLU, message_on_pass=True)
test_predict(input_class=NeuralNetwork, relu_class=ReLU, message_on_pass=True)
test_calculate_multiclass_cross_entropy_loss(input_function=calculate_multiclass_cross_entropy, message_on_pass=True)
test_calculate_accuracy(input_function=calculate_accuracy, message_on_pass=True)

# Last inn data
print("\nLaster MNIST-data...")
data = load_mnist_data(scale_x_data=True)
x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]
x_test, y_test = data["x_test"], data["y_test"]
print(f"Treningssett: {x_train.shape}, Valideringssett: {x_val.shape}, Testsett: {x_test.shape}")

# Vis noen bilder
plot_random_mnist_images(images=x_train, labels=y_train, n_random=10, title="Tilfeldige MNIST-bilder")

# Bygg og tren nettverket
n_features = x_train.shape[1]  # 784
n_classes = 10

model = NeuralNetwork(
    layer_sizes=[n_features, 64, n_classes],
    activation_functions=[Sigmoid(), Sigmoid()],
)
print(f"\nAntall parametere: {model.count_parameters()}")

model.train(
    x_train=x_train,
    y_train=y_train,
    eta=0.1,
    n_epochs=10,
    loss_func=calculate_multiclass_cross_entropy,
    accuracy_func=calculate_accuracy,
    minibatch_size=64,
    eval_set=(x_val, y_val),
)

# Evaluer på testsett
test_preds = model.predict(x_test)
test_acc = calculate_accuracy(y_test, test_preds)
print(f"\nTestsett-nøyaktighet: {test_acc:.4f}")

# Vis feilklassifiserte bilder
plot_mislabeled_mnist_images(
    images=x_test,
    predictions=test_preds,
    labels=y_test,
    n_random=10,
    title="Feilklassifiserte bilder",
)
