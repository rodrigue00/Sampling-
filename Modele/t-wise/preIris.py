import pandas as pd
import numpy as np
from sklearn import datasets
from allpairspy import AllPairs

# CHARGEMENT DU DATASET IRIS
iris_dataset = datasets.load_iris()
iris_df_original = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)  # Dataset original
iris_df_original["target"] = iris_dataset.target  # Ajouter la colonne de classe

# DÉFINITION AUTOMATIQUE DES INTERVALLES (BINS)
bins_dict = {
    "sepal length (cm)": [4.3, 5.3, 6.9, 7.9],  # Catégories pour la longueur des sépales
    "sepal width (cm)": [2.0, 4.4],              # Une seule catégorie
    "petal length (cm)": [1.0, 1.9, 7.0],         # Deux catégories
    "petal width (cm)": [0.1, 0.7, 2.6]           # Deux catégories
}

#  COPIER LE DATASET ORIGINAL ET APPLIQUER LA CATÉGORISATION
iris_df_categorized = iris_df_original.copy()  # Copie pour ne pas modifier l'original

for feature, bins in bins_dict.items():
    labels = list(range(1, len(bins)))  # Création automatique des labels (1, 2, …)
    iris_df_categorized[feature] = pd.cut(iris_df_original[feature], bins=bins, labels=labels, include_lowest=True)

# PRÉPARATION DES VALEURS DES PARAMÈTRES POUR allpairspy, y COMPRIS LA TARGET
# On inclut ici toutes les colonnes catégorisées et la colonne target
parameter_values = {
    **{feature: iris_df_categorized[feature].dropna().unique().tolist() for feature in bins_dict.keys()},
    "target": iris_df_categorized["target"].dropna().unique().tolist()
}

# GÉNÉRATION DES COMBINAISONS OPTIMISÉES (Pairwise Testing)
pairs = list(AllPairs(parameter_values.values()))

# Création d'un DataFrame avec les combinaisons réduites
df_pairwise = pd.DataFrame(pairs, columns=parameter_values.keys())

# AFFICHAGE DU DATASET OPTIMISÉ AVEC LA COLONNE target INCLUSE
print(" Dataset optimisé avec Pairwise Testing incluant la target :")
print(df_pairwise)
