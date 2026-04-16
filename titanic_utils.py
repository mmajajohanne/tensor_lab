import numpy as np
from seaborn import load_dataset
from sklearn.model_selection import train_test_split

RANDOM_STATE = 1160
np.random.seed(RANDOM_STATE)

FEATURES = ["pclass", "sex", "age", "fare", "embarked", "sibsp", "parch"]


def load_titanic_data(test_size=0.2):
    """Laster Titanic-datasettet, gjør om kategoriske verdier og fyller inn manglende verdier."""
    df = load_dataset("titanic")

    df = df[FEATURES + ["survived"]].copy()

    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2})

    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    x = df[FEATURES].to_numpy()
    y = df["survived"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=RANDOM_STATE
    )

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": FEATURES,
    }
