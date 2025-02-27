# Réimportation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from numpy import random
import pandas as pd

# Rechargement du dataset Iris
iris = datasets.load_iris()
X = iris.data  # Utilisation de toutes les features
y = iris.target

# Définition des couleurs et labels pour les catégories
colors = ['red', 'green', 'blue']
labels = ['Setosa', 'Versicolor', 'Virginica']

# Définition des paramètres à tester
n_components_range = [1, 2, 3, 4, 5]
covariance_types = ['full', 'tied', 'diag', 'spherical']

# Définition du taux de réduction
reduction_rate = 0.5  # Générer 50% du nombre d'instances réelles

# Listes pour stocker les nouveaux datasets optimisés
X_new_optimized_list = []
y_new_optimized_list = []

# Récupération des noms de colonnes depuis le dataset Iris d'origine
feature_names = iris.feature_names

# Calcul du nombre total de points à générer
total_samples = int(len(X) * reduction_rate)

# Création de la figure pour la visualisation
plt.figure(figsize=(18, 12))

# Parcours des catégories (Setosa, Versicolor, Virginica)
for idx, (category, color, label) in enumerate(zip(range(3), colors, labels), start=1):
    # Isolation de la catégorie
    X_cat = X[y == category]

    # Initialisation des variables pour le meilleur modèle
    best_bic = np.inf
    best_gmm = None
    best_params = {}

    # Recherche manuelle des meilleurs hyperparamètres
    for n_components in n_components_range:
        for covariance_type in covariance_types:
            gmm = GaussianMixture(n_components=n_components, 
                                  covariance_type=covariance_type, 
                                  random_state=42, 
                                  max_iter=100,  
                                  n_init=3)
            gmm.fit(X_cat)
            bic = gmm.bic(X_cat)

            # Mise à jour du meilleur modèle si BIC plus bas
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_params = {
                    'n_components': n_components,
                    'covariance_type': covariance_type
                }

    # Affichage des meilleurs paramètres
    print(f"\n{label} - Meilleurs paramètres trouvés : {best_params}")
    print(f"{label} - Meilleur score BIC : {best_bic}")

    # Calcul du nombre de points à générer de manière équilibrée avec une légère variation aléatoire
    np.random.seed(None)  # Permet un échantillonnage aléatoire à chaque exécution
    n_samples = total_samples // 3 + np.random.randint(-0, 3)  # Répartition équilibrée avec une variation de ±2
    print(f"{label} - Nombre de points générés (équilibré) : {n_samples}")

    # Génération de points à partir du modèle GMM optimisé
    X_generated_cat, y_generated_cat = best_gmm.sample(n_samples)

    # Arrondi à 1 décimale après la virgule
    X_generated_cat = np.round(X_generated_cat, 1)

    # Stockage des nouveaux datasets optimisés
    X_new_optimized_list.append(X_generated_cat)
    y_new_optimized_list.append(np.full(n_samples, category))

    # Ajout de jitter (petit décalage aléatoire) pour mieux voir les superpositions
    jitter_strength = 0.02
    X_cat_jitter = X_cat + random.uniform(-jitter_strength, jitter_strength, X_cat.shape)

    # Visualisation sur Longueur et largeur du sépale
    plt.subplot(2, 3, idx)
    plt.scatter(X_cat_jitter[:, 0], X_cat_jitter[:, 1],  
                color=color,  
                label=f'{label} (réel)',  
                alpha=0.9,  
                s=60,       
                edgecolor='k',  
                marker='o') 
    plt.scatter(X_generated_cat[:, 0], X_generated_cat[:, 1],  
                color=color,  
                label=f'{label} (généré)', 
                alpha=0.4, 
                marker='x') 
    centers_cat = best_gmm.means_
    plt.scatter(centers_cat[:, 0], centers_cat[:, 1],  
                c='black',  
                s=200,  
                marker='X',  
                label='Centres des clusters')
    plt.title(f"{label} - Sépale (Optimisé)\nTotal généré : {n_samples}")
    plt.xlabel("Longueur du sépale (cm)")
    plt.ylabel("Largeur du sépale (cm)")
    plt.legend()

    # Visualisation sur Longueur et largeur du pétale
    plt.subplot(2, 3, idx + 3)
    plt.scatter(X_cat_jitter[:, 2], X_cat_jitter[:, 3],  
                color=color,  
                label=f'{label} (réel)',  
                alpha=0.9,  
                s=60,       
                edgecolor='k',  
                marker='o') 
    plt.scatter(X_generated_cat[:, 2], X_generated_cat[:, 3],  
                color=color,  
                label=f'{label} (généré)', 
                alpha=0.4, 
                marker='x') 
    centers_cat_petal = best_gmm.means_
    plt.scatter(centers_cat_petal[:, 2], centers_cat_petal[:, 3],  
                c='black',  
                s=200,  
                marker='X',  
                label='Centres des clusters')
    plt.title(f"{label} - Pétale (Optimisé)\nTotal généré : {n_samples}")
    plt.xlabel("Longueur du pétale (cm)")
    plt.ylabel("Largeur du pétale (cm)")
    plt.legend()

plt.tight_layout()
plt.show()

# Création du nouveau dataset optimisé en concaténant les listes
X_new_optimized = np.vstack(X_new_optimized_list)
y_new_optimized = np.hstack(y_new_optimized_list)

# Création du DataFrame pour le nouveau dataset optimisé avec les noms d'origine
df_new_optimized = pd.DataFrame(X_new_optimized, columns=feature_names)
df_new_optimized['species'] = y_new_optimized

# Affichage du nouveau dataset optimisé avec features et target
print("\nNouveau dataset optimisé avec features et target (noms d'origine) :")
print(df_new_optimized.head(75))

# Vérification de la répartition des catégories dans la nouvelle target
print("\nRépartition des catégories dans le nouveau dataset optimisé :")
print(df_new_optimized['species'].value_counts())

# Optionnel : Enregistrer le nouveau dataset optimisé en CSV avec les noms de colonnes d'origine
df_new_optimized.to_csv('nouveau_dataset_iris_optimise_equilibre.csv', index=False)
print("\nLe nouveau dataset optimisé a été enregistré en tant que 'nouveau_dataset_iris_optimise_equilibre.csv' avec les noms de colonnes d'origine.")
