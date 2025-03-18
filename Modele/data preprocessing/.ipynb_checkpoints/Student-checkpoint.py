import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(file_path, output_file):
    """
    Fonction pour effectuer le preprocessing du dataset.
    - Supprime la colonne 'id'
    - Encode les variables catégorielles
    - Génère un nouveau fichier CSV avec les données prétraitées

    Paramètres:
    - file_path (str): Chemin du fichier CSV d'origine
    - output_file (str): Chemin du fichier CSV de sortie (prétraité)
    """
    # Charger le dataset CSV avec gestion des erreurs
    try:
        df = pd.read_csv(file_path)
        print(" Données chargées avec succès.")
    except FileNotFoundError:
        raise FileNotFoundError(f" Le fichier '{file_path}' est introuvable.")

    # Définition de la colonne cible
    target_column = "Depression"

    # Vérification si la colonne cible existe dans le DataFrame
    if target_column not in df.columns:
        raise ValueError(f" La colonne cible '{target_column}' est absente des données.")
    print(f" Colonne cible définie : {target_column}")

    # Suppression de la colonne 'id' qui est généralement inutile pour l'analyse
    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)
        print(" Colonne 'id' supprimée.")

    # Encodage de la variable catégorielle 'Gender' en variables binaires
    if 'Gender' in df.columns:
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
        print(" Encodage de 'Gender' effectué.")

    # Encodage des variables catégorielles ordinales en valeurs numériques
    enc = LabelEncoder()
    ordinal_columns = ['City', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Financial Stress']
    for col in ordinal_columns:
        if col in df.columns:
            df[col] = enc.fit_transform(df[col])
            print(f"Encodage de '{col}' effectué.")

    # Encodage des variables binaires sous forme de variables indicatrices
    binary_columns = ['Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']
    for col in binary_columns:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            print(f"Encodage binaire de '{col}' effectué.")

    # Sauvegarde du DataFrame prétraité dans un nouveau fichier CSV
    df.to_csv(output_file, index=False)
    print(f"\n Nouveau fichier CSV enregistré sous '{output_file}'.")

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemin du fichier d'origine et du fichier de sortie
    input_file = "../data/Student.csv"
    output_file = "../data/preprocessed_Student.csv"
    
    # Appel de la fonction de preprocessing
    preprocess_dataset(input_file, output_file)
