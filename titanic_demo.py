import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from titanic_utils import RANDOM_STATE, load_titanic_data

# Last inn data
data = load_titanic_data()
x_train = data["x_train"]
x_test = data["x_test"]
y_train = data["y_train"]
y_test = data["y_test"]
feature_names = data["feature_names"]

print(f"Treningssett: {x_train.shape}, Testsett: {x_test.shape}")


# --- Beslutningstre ---

def visualize_tree(decision_tree, column_names, max_depth=3):
    plt.figure(figsize=(16, 8))
    plot_tree(
        decision_tree,
        max_depth=max_depth,
        feature_names=column_names,
        class_names=["døde", "overlevde"],
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title(f"Beslutningstre (dybde {decision_tree.get_depth()})")
    plt.tight_layout()
    plt.show()


# Ubegrenset tre
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree.fit(x_train, y_train)

print(f"\nUbegrenset beslutningstre:")
print(f"  Dybde: {tree.get_depth()}, Løvnoder: {tree.get_n_leaves()}")
print(f"  Treningsnøyaktighet: {accuracy_score(y_train, tree.predict(x_train)):.4f}")
print(f"  Testnøyaktighet:     {accuracy_score(y_test, tree.predict(x_test)):.4f}")

visualize_tree(tree, feature_names)

# Regularisering med ulike dybder
tree_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, None]

print(f"\n{'Dybde':<8} {'Løvnoder':<12} {'Trenings-nøy.':<16} {'Test-nøy.'}")
print("-" * 50)
for depth in tree_depths:
    t = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    t.fit(x_train, y_train)
    train_acc = accuracy_score(y_train, t.predict(x_train))
    test_acc = accuracy_score(y_test, t.predict(x_test))
    depth_str = str(depth) if depth is not None else "None"
    print(f"{depth_str:<8} {t.get_n_leaves():<12} {train_acc:<16.4f} {test_acc:.4f}")


# --- Kryssvalidering ---

# Ubegrenset tre med kryssvalidering
scores = cross_val_score(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    x_train, y_train, cv=5, scoring="accuracy"
)
print(f"\nKryssvalidering (ubegrenset tre):")
print(f"  Gjennomsnitt: {scores.mean():.4f}, Standardavvik: {scores.std():.4f}")

# Finn beste dybde med kryssvalidering
print(f"\n{'Dybde':<8} {'Kryssval.-nøy.'}")
print("-" * 25)
best_depth, best_cv_score = None, 0
for depth in tree_depths:
    cv_scores = cross_val_score(
        DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE),
        x_train, y_train, cv=5, scoring="accuracy"
    )
    mean_score = cv_scores.mean()
    depth_str = str(depth) if depth is not None else "None"
    print(f"{depth_str:<8} {mean_score:.4f}")
    if mean_score > best_cv_score:
        best_cv_score = mean_score
        best_depth = depth

print(f"\nBeste dybde: {best_depth} (kryssval.-nøy. {best_cv_score:.4f})")

best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
best_tree.fit(x_train, y_train)
print(f"  Treningsnøyaktighet: {accuracy_score(y_train, best_tree.predict(x_train)):.4f}")
print(f"  Testnøyaktighet:     {accuracy_score(y_test, best_tree.predict(x_test)):.4f}")


# --- Tilfeldig skog ---

forest = RandomForestClassifier(n_estimators=100, max_depth=best_depth, random_state=RANDOM_STATE)
forest.fit(x_train, y_train)
print(f"\nTilfeldig skog (100 trær, dybde={best_depth}):")
print(f"  Treningsnøyaktighet: {accuracy_score(y_train, forest.predict(x_train)):.4f}")
print(f"  Testnøyaktighet:     {accuracy_score(y_test, forest.predict(x_test)):.4f}")

# Finn best antall trær med kryssvalidering
n_trees_list = [1, 5, 10, 20, 35, 50, 75, 100, 200, 500, 1000]

print(f"\n{'Antall trær':<14} {'Kryssval.-nøy.'}")
print("-" * 30)
best_n_trees, best_forest_score = 100, 0
for n in n_trees_list:
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=n, max_depth=best_depth, random_state=RANDOM_STATE),
        x_train, y_train, cv=5, scoring="accuracy"
    )
    mean_score = cv_scores.mean()
    print(f"{n:<14} {mean_score:.4f}")
    if mean_score > best_forest_score:
        best_forest_score = mean_score
        best_n_trees = n

print(f"\nBeste antall trær: {best_n_trees} (kryssval.-nøy. {best_forest_score:.4f})")

best_forest = RandomForestClassifier(n_estimators=best_n_trees, max_depth=best_depth, random_state=RANDOM_STATE)
best_forest.fit(x_train, y_train)
print(f"  Treningsnøyaktighet: {accuracy_score(y_train, best_forest.predict(x_train)):.4f}")
print(f"  Testnøyaktighet:     {accuracy_score(y_test, best_forest.predict(x_test)):.4f}")


# --- k-NN ---

k_values = [1, 3, 5, 7, 10, 15, 20, 30, 50, 100]

print(f"\n{'k':<6} {'Kryssval.-nøy.'}")
print("-" * 22)
best_k, best_knn_score = 1, 0
for k in k_values:
    cv_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=k),
        x_train, y_train, cv=5, scoring="accuracy"
    )
    mean_score = cv_scores.mean()
    print(f"{k:<6} {mean_score:.4f}")
    if mean_score > best_knn_score:
        best_knn_score = mean_score
        best_k = k

print(f"\nBeste k: {best_k} (kryssval.-nøy. {best_knn_score:.4f})")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(x_train, y_train)
print(f"  Treningsnøyaktighet: {accuracy_score(y_train, best_knn.predict(x_train)):.4f}")
print(f"  Testnøyaktighet:     {accuracy_score(y_test, best_knn.predict(x_test)):.4f}")


# --- Nevralt nettverk ---

hidden_sizes = [
    (5,),
    (10,),
    (50,),
    (100,),
    (25, 25),
    (50, 50),
    (25, 100, 25),
]

print(f"\n{'Lag':<16} {'Trenings-nøy.':<16} {'Test-nøy.'}")
print("-" * 45)
best_mlp_config, best_mlp_test_acc = (50,), 0
for sizes in hidden_sizes:
    mlp = MLPClassifier(hidden_layer_sizes=sizes, random_state=RANDOM_STATE, max_iter=500)
    mlp.fit(x_train, y_train)
    train_acc = accuracy_score(y_train, mlp.predict(x_train))
    test_acc = accuracy_score(y_test, mlp.predict(x_test))
    print(f"{str(sizes):<16} {train_acc:<16.4f} {test_acc:.4f}")
    if test_acc > best_mlp_test_acc:
        best_mlp_test_acc = test_acc
        best_mlp_config = sizes

print(f"\nBeste MLP-konfigurasjon: {best_mlp_config} (test-nøy. {best_mlp_test_acc:.4f})")

best_mlp = MLPClassifier(hidden_layer_sizes=best_mlp_config, random_state=RANDOM_STATE, max_iter=500)
best_mlp.fit(x_train, y_train)


# --- Ensemble (voting) ---

ensemble = VotingClassifier(estimators=[
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ("mlp", MLPClassifier(hidden_layer_sizes=best_mlp_config, random_state=RANDOM_STATE, max_iter=500)),
    ("forest", RandomForestClassifier(n_estimators=best_n_trees, max_depth=best_depth, random_state=RANDOM_STATE)),
])
ensemble.fit(x_train, y_train)

print(f"\nEnsemble (k-NN + MLP + tilfeldig skog):")
print(f"  Treningsnøyaktighet: {accuracy_score(y_train, ensemble.predict(x_train)):.4f}")
print(f"  Testnøyaktighet:     {accuracy_score(y_test, ensemble.predict(x_test)):.4f}")

# Sammenligning
print(f"\n{'Modell':<30} {'Testnøyaktighet'}")
print("-" * 45)
for name, model in [
    ("Beste beslutningstre", best_tree),
    ("Tilfeldig skog", best_forest),
    (f"k-NN (k={best_k})", best_knn),
    (f"MLP {best_mlp_config}", best_mlp),
    ("Ensemble", ensemble),
]:
    print(f"{name:<30} {accuracy_score(y_test, model.predict(x_test)):.4f}")
