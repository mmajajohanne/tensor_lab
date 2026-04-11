import numpy as np
import seaborn as sns
from openml.datasets import get_dataset
from sklearn.preprocessing import StandardScaler


def _extract_columns_from_data_by_name(data, feature_names, columns_to_include):
    try:
        indices_to_include = [feature_names.index(column_name) for column_name in columns_to_include]
    except ValueError:
        raise ValueError(f"All values in `columns_to_include` must be in {feature_names}. Got {columns_to_include}. ")
    return data[:, indices_to_include]


def _split_data_in_train_val_test(x_data, y_data, val_ratio=0.2, test_ratio=0.2, seed=57):
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError(
            f"Arguments `val_ratio` and `test_ratio` must be between 0 and 1. Got {val_ratio=} and {test_ratio=}. "
        )
    if val_ratio + test_ratio >= 1:
        raise ValueError(
            f"Arguments `val_ratio` and `test_ratio` must be less than 1 summed. Got {val_ratio=} and {test_ratio=}. "
        )
    rng = np.random.default_rng(seed=seed)
    n = x_data.shape[0]
    random_indices = rng.permutation(n)

    train_ratio = 1 - val_ratio - test_ratio
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_indices = random_indices[:n_train]
    val_indices = random_indices[n_train : n_train + n_val]
    test_indices = random_indices[n_train + n_val :]

    return {
        "x_train": x_data[train_indices],
        "x_val": x_data[val_indices],
        "x_test": x_data[test_indices],
        "y_train": y_data[train_indices],
        "y_val": y_data[val_indices],
        "y_test": y_data[test_indices],
    }


def _scale_data_splits(data_dict):
    scaler = StandardScaler()
    data_dict["x_train"] = scaler.fit_transform(data_dict["x_train"])
    data_dict["x_val"] = scaler.transform(data_dict["x_val"])
    data_dict["x_test"] = scaler.transform(data_dict["x_test"])
    return data_dict


def get_auto_mpg_data(columns_to_include=None, perform_scaling=True, val_ratio=0.2, test_ratio=0.2, seed=57):
    df = sns.load_dataset("mpg").dropna()
    all_feature_names = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
    if columns_to_include is None:
        columns_to_include = all_feature_names

    x_data = df[columns_to_include].to_numpy()
    y_data = df["mpg"].to_numpy()

    filtered_x_data = _extract_columns_from_data_by_name(
        data=x_data, feature_names=columns_to_include, columns_to_include=columns_to_include
    )

    all_data = _split_data_in_train_val_test(
        x_data=filtered_x_data, y_data=y_data, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    if perform_scaling:
        all_data = _scale_data_splits(data_dict=all_data)
    all_data["feature_names"] = columns_to_include
    all_data["target_name"] = "mpg"
    return all_data


def get_spambase_data(columns_to_include=None, perform_scaling=True, val_ratio=0.2, test_ratio=0.2, seed=57):
    spam_metadata = get_dataset(44, version=1)
    x_data, y_data, _, all_feature_names = spam_metadata.get_data(target="class")

    if columns_to_include is None:
        columns_to_include = all_feature_names

    x_data = x_data[columns_to_include].to_numpy()
    y_data = y_data.astype(int).to_numpy()

    all_data = _split_data_in_train_val_test(
        x_data=x_data, y_data=y_data, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    if perform_scaling:
        all_data = _scale_data_splits(data_dict=all_data)
    all_data["feature_names"] = columns_to_include
    all_data["target_name"] = "spam"
    return all_data
