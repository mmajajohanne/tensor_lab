import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


def plot_single_mnist_image(image, pred=None, label=None, show=True, title=None):
    """Viser ett enkelt MNIST-bilde."""
    image = np.reshape(image, (28, 28))
    plt.imshow(image, cmap="gray")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    if title is None:
        title = ""
    else:
        title += ". "
    if pred is not None:
        title += f"Prediksjon: {pred}"
    if pred is not None and label is not None:
        title += ", "
    if label is not None:
        title += f"Sann klasse: {label}"
    plt.title(title)
    if show:
        plt.show()


def plot_mnist_images(images, predictions=None, labels=None, show=True, n_images=10, n_cols=5, title=None):
    """Viser flere MNIST-bilder i et rutenett."""
    fig = plt.figure()
    n_rows = int(np.ceil(n_images / n_cols))
    for i in range(n_images):
        if predictions is not None and labels is not None:
            if predictions[i] != labels[i]:
                rc("text", color="blue")
            else:
                rc("text", color="black")

        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        image = np.reshape(images[i], (28, 28))
        ax.imshow(image, cmap="gray")

        sub_title = ""
        if predictions is not None:
            sub_title += f"P: {predictions[i]}"
        if predictions is not None and labels is not None:
            sub_title += ", "
        if labels is not None:
            sub_title += f"Y: {labels[i]}"
        ax.title.set_text(sub_title)

        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    rc("text", color="black")
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()


def plot_random_mnist_images(
    images, predictions=None, labels=None, show=True, n_random=10, n_cols=5, title=None, seed=57
):
    """Viser tilfeldige MNIST-bilder."""
    n = images.shape[0]
    np.random.seed(seed=seed)
    indices = np.random.choice(n, n_random, replace=False)
    random_images = images[indices]
    if predictions is not None:
        predictions = predictions[indices]
    if labels is not None:
        labels = labels[indices]
    plot_mnist_images(
        images=random_images,
        predictions=predictions,
        labels=labels,
        show=show,
        n_images=n_random,
        n_cols=n_cols,
        title=title,
    )


def plot_mislabeled_mnist_images(images, predictions, labels, show=True, n_random=10, n_cols=5, title=None, seed=57):
    """Viser tilfeldige feilklassifiserte MNIST-bilder."""
    indices = predictions != labels
    plot_random_mnist_images(
        images=images[indices],
        predictions=predictions[indices],
        labels=labels[indices],
        show=show,
        n_random=n_random,
        n_cols=n_cols,
        title=title,
        seed=seed,
    )


def plot_worst_predicted_mnist_images(images, logits, labels, show=True, n_images=10, n_cols=5, title=None):
    """Viser bildene med dårligst prediksjoner basert på logit-verdier for sann klasse."""
    predicted_logits = logits[np.arange(len(labels)), labels]
    indices = predicted_logits.argsort()
    worst_images = images[indices[:n_images]]
    worst_logits = logits[indices[:n_images]]
    worst_labels = labels[indices[:n_images]]
    predictions = worst_logits.argmax(axis=1)
    plot_mnist_images(
        images=worst_images,
        predictions=predictions,
        labels=worst_labels,
        show=show,
        n_images=n_images,
        n_cols=n_cols,
        title=title,
    )
