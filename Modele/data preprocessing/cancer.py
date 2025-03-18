import numpy as np
import pandas as pd


"""
 Description du jeu de Données "données_sur_le_risque_de_cancer_de_la_thyroide"
Structure du Jeu de Données

Le jeu de données comprend 17 colonnes et 212691 lignes.  décrites ci-dessous:
Patient_ID : Identifiant unique du patient.
Age : Âge du patient en années.
Gender : Sexe du patient (Male ou Female).
Country : Pays d'origine du patient.
Ethnicity : Ethnicité du patient (Caucasian, Hispanic, Asian, African).
Family_History : Antécédents familiaux de cancer de la thyroïde (Yes ou No).
Radiation_Exposure : Exposition à des radiations (Yes ou No).
Iodine_Deficiency : Carence en iode (Yes ou No).
Smoking : Consommation de tabac (Yes ou No).
Obesity : Obésité (Yes ou No).
Diabetes : Présence de diabète (Yes ou No).
TSH_Level : Niveau de l'hormone stimulant la thyroïde (TSH).
T3_Level : Niveau de triiodothyronine (T3).
T4_Level : Niveau de thyroxine (T4).
Nodule_Size : Taille du nodule thyroïdien en centimètres.
Thyroid_Cancer_Risk : Niveau de risque estimé du cancer de la thyroïde (Low, Medium, High).
Diagnosis : Diagnostic final (Benign ou Malignant)
    """

# Charger le dataset fourni
file_path = 'data/thyroid_cancer_risk_data.csv'
df = pd.read_csv(file_path)
print(" Données chargées avec succès.")

# Suppression de la colonne 'patient_id'
if 'Patient_ID' in df.columns:
    df = df.drop('Patient_ID', axis=1)
    print("Colonne 'patient_id' supprimée.")

# Normalisation des noms de colonnes : remplacement des espaces par des underscores et conversion en minuscules
df.columns = df.columns.str.replace(" ", "_").str.lower()
print(" Noms de colonnes normalisés :")
print(df.columns)

# Vérification du nombre de valeurs manquantes dans chaque colonne
print("\n Valeurs manquantes par colonne :")
print(df.isna().sum())

# Sélection des colonnes de type numérique (int et float)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print("\n Colonnes numériques :")
print(numeric_columns)

# Définition des colonnes catégorielles
categorical_cols = ['gender', 'country', 'ethnicity', 'family_history', 'radiation_exposure', 
                    'iodine_deficiency', 'smoking', 'obesity', 'diabetes', 'thyroid_cancer_risk', 
                    'diagnosis']

# Affichage des valeurs uniques pour chaque colonne catégorielle (avant encodage)
print("\n Valeurs uniques des colonnes catégorielles :")
for col in categorical_cols:
    if col in df.columns:
        print(f"{col} : {df[col].unique()}")

# Transformation des valeurs de la colonne 'diagnosis' en format numérique
if 'diagnosis' in df.columns:
    df['diagnosis'] = df['diagnosis'].map({'Benign': 0, 'Malignant': 1})
    print("\n Encodage de 'diagnosis' effectué.")

# Transformation des niveaux de risque du cancer thyroïdien en valeurs numériques ordonnées
if 'thyroid_cancer_risk' in df.columns:
    df['thyroid_cancer_risk'] = df['thyroid_cancer_risk'].map({'Low': 1, 'Medium': 2, 'High': 3})
    print("\n Encodage de 'thyroid_cancer_risk' effectué.")

# Liste des colonnes catégorielles nominales (sans ordre particulier)
nominal_cols = ['gender', 'country', 'ethnicity', 'family_history', 'radiation_exposure', 
                'iodine_deficiency', 'smoking', 'obesity', 'diabetes']

# Encodage des variables catégorielles nominales en variables binaires (one-hot encoding)
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
print("\n Encodage One-Hot des colonnes nominales effectué.")

df['Age_Category'] = pd.cut(df['Age'], 
                            bins=[0, 15, 35, 50, 65, float('inf')], 
                            labels=[0, 1, 2, 3, 4])

# Définition de la colonne cible pour la modélisation
target_column = "diagnosis"

# Vérification de la présence de la colonne cible dans les données
if target_column not in df.columns:
    raise ValueError(f" La colonne cible '{target_column}' est absente des données.")
print(f" Colonne cible définie : {target_column}")

# Sauvegarde du DataFrame prétraité dans un nouveau fichier CSV
output_file = 'data/preprocessed_thyroid_cancer_risk_dataset.csv'
df.to_csv(output_file, index=False)
print(f"\n Nouveau fichier CSV enregistré sous '{output_file}'.")

# Affichage des premières lignes du dataset prétraité
df.head()
