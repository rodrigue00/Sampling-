
"""
    Description du jeu de Données "winequality-red"

Structure du Jeu de Données :
•	Nombre de Colonnes : 12
•	Nombre lignes : 1599
________________________________________
Attributs :
1.	fixed acidity (Acidité Fixe) : Concentration des acides non volatils (exprimée en g/dm³).
2.	volatile acidity (Acidité Volatile) : Concentration des acides volatils responsables des arômes vinaigrés (g/dm³).
3.	citric acid (Acide Citrique) : Indicateur de fraîcheur et de saveur fruitée (g/dm³).
4.	residual sugar (Sucre Résiduel) : Quantité de sucre restant après fermentation (g/dm³).
5.	chlorides (Chlorures) : Teneur en sel (g/dm³).
6.	free sulfur dioxide (Dioxyde de Soufre Libre) : SO₂ sous forme libre, protège contre l'oxydation (mg/dm³).
7.	total sulfur dioxide (Dioxyde de Soufre Total) : SO₂ total (libre et combiné) dans le vin (mg/dm³).
8.	density (Densité) : Masse volumique du vin (g/cm³), proche de celle de l'eau.
9.	pH : Niveau d'acidité du vin (sans unité, échelle logarithmique).
10.	sulphates (Sulfates) : Indicateur de la préservation microbiologique (g/dm³).
11.	alcohol (Teneur en Alcool) : Pourcentage d'alcool par volume (% vol).
12.	quality (Qualité) : Note attribuée par des experts (de 0 à 10).


    """
import pandas as pd

file_path = 'data/winequality-red.csv'
df = pd.read_csv(file_path)
print(" Données chargées avec succès.")

# Étape 1 : Vérification des valeurs manquantes
print("\n Vérification des valeurs manquantes :")
missing_data = df.isnull().sum()
print(missing_data)

# Vérification des nouvelles colonnes
print("\n Aperçu des premières lignes du DataFrame après transformation :")
print(df.head())

# Quality classes :
df.quality.unique()

# Étape 1 : transformer les notes en labels textuels
df['quality'] = df['quality'].apply(lambda x: 'Good' if x >= 6 else 'Bad')

# Étape 2 : remplacer Good → 1, Bad → 0
df['quality'] = df['quality'].replace({'Good': 1, 'Bad': 0})

# Définition de la colonne cible pour la modélisation
target_column = "quality"

# Vérification de la présence de la colonne cible dans les données
if target_column not in df.columns:
    raise ValueError(f" La colonne cible '{target_column}' est absente des données.")
print(f" Colonne cible définie : {target_column}")

# Sauvegarde du DataFrame prétraité dans un nouveau fichier CSV
output_file = 'data/preprocessed_winequality_dataset.csv'
df.to_csv(output_file, index=False)
print(f"\n Nouveau fichier CSV enregistré sous '{output_file}'.")


