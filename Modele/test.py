import matplotlib.pyplot as plt
import pandas as pd

# Définition du chemin du fichier
file_path = "resultats_iterations.xlsx"

# Charger la feuille de calcul dans un DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Création de boxplots simplifiés en noir et blanc
plt.figure(figsize=(8, 4))

# Boxplot pour Accuracy Avant
plt.subplot(1, 2, 1)
plt.boxplot([df['Accuracy Avant'][df['Modèle'] == model] for model in df['Modèle'].unique()])
plt.xticks(range(1, len(df['Modèle'].unique()) + 1), df['Modèle'].unique(), rotation=45)
plt.title("Accuracy Avant")

# Boxplot pour Accuracy Après
plt.subplot(1, 2, 2)
plt.boxplot([df['Accuracy Après'][df['Modèle'] == model] for model in df['Modèle'].unique()])
plt.xticks(range(1, len(df['Modèle'].unique()) + 1), df['Modèle'].unique(), rotation=45)
plt.title("Accuracy Après")

# Ajustement et affichage
plt.tight_layout()
plt.show()
