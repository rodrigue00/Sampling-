# Importation des librairies nécessaires
import numpy as np
import pandas as pd
from sklearn import datasets
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Fonction pour générer les combinaisons N-wise
def generate_n_wise(X, n):
    print(f"\n--- Génération des combinaisons {n}-wise ---")
    n_wise_features = list(combinations(X.columns, n))
    n_wise_list = []

    for index, row in X.iterrows():
        n_wise = []
        for combination in n_wise_features:
            values = tuple(row[feat] for feat in combination)
            n_wise.append(values)
        n_wise_list.append(n_wise)

    print(f"Total des combinaisons {n}-wise générées : {len(n_wise_list)}")
    return n_wise_list

# Fonction pour sélectionner le sous-ensemble minimal couvrant toutes les combinaisons N-wise
def select_minimal_n_wise(n_wise_list):
    print("\n--- Sélection du sous-ensemble minimal couvrant toutes les combinaisons ---")
    covered_combinations = set()
    selected_indices = []
    selected_combinations = []

    for idx, combinations in enumerate(n_wise_list):
        new_combinations = set(combinations) - covered_combinations
        
        if new_combinations:
            selected_indices.append(idx)
            covered_combinations.update(new_combinations)
            selected_combinations.append((idx, new_combinations))

    print("\n--- Combinaisons sélectionnées ---")
    for index, comb in selected_combinations:
        print(f"Index {index} : {comb}")

    print(f"\nTotal des indices sélectionnés : {len(selected_indices)}")
    return selected_indices

# Fonction principale pour tester automatiquement avec différents N-wise et 6 quantiles
def auto_n_wise(q=6):
    print("\n--- Début des tests N-wise automatiques avec 6 quantiles ---")

    # Charger le dataset Iris
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Affichage du dataset original
    print("\n--- Dataset original (avant discrétisation) ---")
    print(X.head())
    print("\n--- Distribution des classes (avant réduction) ---")
    print(pd.Series(y).value_counts())

    # Nombre maximum de features (limite de N-wise)
    max_features = len(X.columns)

    # Discrétisation optimisée avec 6 quantiles
    print("\n--- Discrétisation optimisée avec 6 quantiles ---")
    for column in X.columns:
        # Récupération des intervalles de quantiles
        quantiles, bins = pd.qcut(X[column], q=q, retbins=True, labels=[f'Q{i+1}' for i in range(q)])
        X[column] = quantiles
        print(f"\nIntervalles pour {column} : {bins}")

        # Affichage des données pour chaque catégorie
        print(f"\nDonnées pour chaque catégorie ({column}) :")
        print(X[column].value_counts().sort_index())

    # Affichage du dataset après discrétisation
    print("\n--- Dataset après discrétisation ---")
    print(X.head())

    # Encodage des catégories en valeurs numériques
    print("\n--- Encodage des catégories en valeurs numériques ---")
    label_encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
        print(f"{column} encodé : {X[column].unique()}")

    # Boucle automatique pour augmenter N-wise
    n = 2  # Commence avec Pairwise (2-way)
    while n <= max_features:
        print(f"\n--- Test avec {n}-wise ---")

        # Génération des combinaisons N-wise
        n_wise_list = generate_n_wise(X, n)

        # Sélection du sous-ensemble minimal couvrant toutes les combinaisons N-wise
        selected_indices = select_minimal_n_wise(n_wise_list)

        # Vérification du nombre d'instances sélectionnées
        if len(selected_indices) == 0:
            print(f"\n {n}-wise n'a généré aucune instance. Ignoré.")
            break

        # Création du dataset réduit avec les indices sélectionnés
        X_reduced = X.iloc[selected_indices]
        y_reduced = y[selected_indices]

        # Vérification de la distribution des classes dans le dataset réduit
        print("\n--- Distribution des classes dans le dataset réduit ---")
        print(pd.Series(y_reduced).value_counts())

        # Affichage du dataset réduit
        print("\n--- Dataset réduit (après sélection N-wise) ---")
        print(X_reduced.head())

        # Split train/test (70% train, 30% test)
        print("\n--- Split train/test (70% train, 30% test) ---")
        X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
            X_reduced, y_reduced, test_size=0.3
        )
        print(f"Nombre d'instances train : {len(y_train_red)}")
        print(f"Nombre d'instances test : {len(y_test_red)}")

        # Entraînement sur le dataset réduit avec N-wise
        print("\n--- Entraînement du modèle Random Forest ---")
        model_reduced = RandomForestClassifier()
        model_reduced.fit(X_train_red, y_train_red)

        # Prédictions et évaluation du modèle réduit
        print("\n--- Prédiction et évaluation du modèle réduit ---")
        y_pred_reduced = model_reduced.predict(X_test_red)
        accuracy = accuracy_score(y_test_red, y_pred_reduced)
        print("\nRapport de classification :")
        print(classification_report(y_test_red, y_pred_reduced))
        print(f"Accuracy : {accuracy}")

        # Augmenter N pour le prochain test
        n += 1

    print("\n--- Fin des tests N-wise automatiques avec 6 quantiles ---")
    print("\n Limite atteinte : Impossible de faire du N-wise avec N > Nombre de features.")
    print(f"Le nombre maximum de features est : {max_features}")

# Lancer les tests N-wise automatiques avec 6 quantiles
auto_n_wise(q=6)
