

import pandas as pd


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
file_path = '../data/winequality-red.csv'
df = pd.read_csv(file_path)
print(" Données chargées avec succès.")

# Étape 1 : Vérification des valeurs manquantes
print("\n Vérification des valeurs manquantes :")
missing_data = df.isnull().sum()
print(missing_data)

# Étape 2 : Création d'une colonne binaire pour la qualité du vin
df['is_good'] = (df['quality'] >= round(df['quality'].mean())).astype(int)
print("\n Création de la colonne binaire 'is_good' effectuée.")

# Suppression de la colonne 'quality' car elle est remplacée par 'is_good'
df.drop(['quality'], axis=1, inplace=True)
print(" Colonne 'quality' supprimée.")

# Vérification des nouvelles colonnes
print("\n Aperçu des premières lignes du DataFrame après transformation :")
print(df.head())

# Sauvegarde du DataFrame prétraité dans un nouveau fichier CSV
output_file = '../data/preprocessed_winequality_dataset.csv'
df.to_csv(output_file, index=False)
print(f"\n Nouveau fichier CSV enregistré sous '{output_file}'.")
