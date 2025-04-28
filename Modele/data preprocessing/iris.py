
import numpy as np
import pandas as pd

"""
    Description du Jeu de Données Iris 
Le jeu de données Iris contient des mesures de 150 fleurs appartenant à trois espèces d'iris:
•	Iris setosa
•	Iris versicolor
•	Iris virginica
Chaque fleur est décrite par quatre caractéristiques :
1.	Longueur du sépale (sepal length) - en centimètres
2.	Largeur du sépale (sepal width) - en centimètres
3.	Longueur du pétale (petal length) - en centimètres
4.	Largeur du pétale (petal width) - en centimètres
Le jeu de données comprend 150 observations, réparties également entre les trois espèces (50 observations par espèce).

    """

# Charger le dataset fourni
file_path = './data/iris.csv'
df = pd.read_csv(file_path)
print(" Données chargées avec succès.")

# Vérification des valeurs uniques de la colonne 'species'
print("\n Valeurs uniques dans la colonne 'species' avant traitement :")
print(df['species'].unique())

# Remplacer les valeurs par des étiquettes numériques (0, 1, 2)
df['species'] = df['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})


# Définition de la colonne cible
target_column = "species"
if target_column not in df.columns:
    raise ValueError(f" La colonne cible '{target_column}' est absente des données.")
print(f" Colonne cible définie : {target_column}")

# Vérification des valeurs après traitement
print("\n Valeurs uniques dans la colonne 'species' après traitement :")
print(df['species'].unique())

# Sauvegarde du DataFrame prétraité dans un nouveau fichier CSV
output_file = './data/preprocessed_iris_dataset.csv'
df.to_csv(output_file, index=False)
print(f"\n Nouveau fichier CSV enregistré sous '{output_file}'.")

# Affichage des premières lignes du dataset prétraité
df.head()

