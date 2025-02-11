import pandas as pd
import json

# 1. Charger la configuration depuis le fichier config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

file_path = config["file_path"]
file_type = config["file_type"].lower()

# 2. Charger les données en fonction du type de fichier
try:
    if file_type == "csv":
        df = pd.read_csv(file_path)  # Charger un fichier CSV
    elif file_type == "json":
        df = pd.read_json(file_path)  # Charger un fichier JSON
    elif file_type == "excel":
        df = pd.read_excel(file_path)  # Charger un fichier Excel
    else:
        raise ValueError("Format de fichier non pris en charge : choisissez 'csv', 'json' ou 'excel'.")
    
    # Afficher les premières lignes des données chargées
    print("Données chargées avec succès :")
    
     # 3. Comprendre les données
     
     # Afficher les premières lignes
    print("\n🔍 Aperçu des données :")
    print(df.head())
    
    # Informations générales sur les données
    df.info()

    # Vérifier la taille du dataset
    print("\n Dimensions du dataset (lignes, colonnes) :", df.shape)

    # Vérifier les types de données
    print("\n Types de données par colonne :")
    print(df.dtypes)

    # Vérifier les valeurs manquantes
    print("\n Valeurs manquantes par colonne :")
    print(df.isnull().sum())
    
    # Vérifier le nombre de valeurs uniques par colonne
    print("\n Nombre de valeurs uniques par colonne :")
    print(df.nunique())
    
    # Détecter les doublons
    print("\n Nombre de lignes dupliquées :", df.duplicated().sum())

    # Statistiques descriptives
    print("\n Statistiques descriptives :")
    print(df.describe())
    
    # Vérifier l’équilibre des classes (si une colonne 'target' existe)
    target_column = "target"  # Modifier si besoin
    if target_column in df.columns:
        print(f"\n Distribution des classes dans '{target_column}' :")
        print(df[target_column].value_counts())

except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' n'existe pas. Vérifiez le chemin.")
except Exception as e:
    print(f"Erreur lors de l'analyse des données : {e}")
