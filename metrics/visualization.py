import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


def scatter_plot(vectors, labels):
    """Gir en 2D-visualisering av vektorrommet."""
    svd = TruncatedSVD(2).fit_transform(vectors)
    x_axis, y_axis = svd[:, 0], svd[:, 1]

    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_to_int[l] for l in labels]

    plt.figure(figsize=(8, 6))
    color_map = plt.cm.get_cmap("jet", len(set(labels)))
    scatter = plt.scatter(x_axis, y_axis, c=numeric_labels, cmap=color_map)
    handles, _ = scatter.legend_elements(prop="colors")
    plt.legend(handles, unique_labels, title="Labels")
    plt.show()
