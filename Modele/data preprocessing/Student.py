"""
   
Description du jeu de Données "Student Depression Dataset"
1.	id : Identifiant unique de l'étudiant
2.	Gender : Genre de l'étudiant (Male/Female)
3.	Age : Âge de l'étudiant (en années)
4.	City : Ville de résidence
5.	Profession : Statut professionnel (Student)
6.	Academic Pressure : Niveau de pression académique (valeurs numériques)
7.	Work Pressure : Niveau de pression professionnelle (valeurs numériques)
8.	CGPA : Moyenne pondérée cumulée de l'étudiant
9.	Study Satisfaction( Satisfaction au travail) : Niveau de satisfaction des études (valeurs numériques)
10.	Job Satisfaction : Niveau de satisfaction professionnelle (valeurs numériques)
11.	Sleep Duration(habitudes alimentaires) : Durée moyenne du sommeil (ex : "5-6 hours", "Less than 5 hours", "7-8 hours")
12.	Dietary Habits : Habitudes alimentaires (Healthy, Moderate, Unhealthy)
13.	Degree : Niveau d'éducation (ex : BSc, BA, PhD, Class 12, etc.)
14.	Have you ever had suicidal thoughts(Avez-vous déjà eu des pensées suicidaires) : Indique si l'étudiant a déjà eu des pensées suicidaires (Yes/No)
15.	Work/Study Hours : Nombre d'heures d'études ou de travail par jour
16.	Financial Stress : Niveau de stress financier (valeurs numériques)
17.	Family History of Mental Illness (Antécédents familiaux de maladie mentale): Indique si l'étudiant a des antécédents familiaux de maladies mentales (Yes/No)
18.	Depression : Indicateur de la présence de dépression (1 pour dépression, 0 pour absence de dépression)
Nombre de Lignes et de Colonnes
Le jeu de données comprend 27901 observations, et 18 colonnes.

    """
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Définition du chemin du fichier
file_path = 'data/Student Depression Dataset.csv'
output_file = 'data/preprocessed_student_depression_dataset.csv'

try:
    # Chargement du dataset
    df = pd.read_csv(file_path)
    print("Données chargées avec succès.")

    # Vérification et suppression de la colonne 'id' si elle existe
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
        print("Colonne 'id' supprimée.")

    # Vérification de la présence de la colonne cible
    target_column = "Depression"
    if target_column not in df.columns:
        raise ValueError(f"La colonne cible '{target_column}' est absente des données.")
    print(f"Colonne cible définie : {target_column}")

    # Encodage de la variable 'Gender'
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        print("Encodage de 'Gender' en 0 et 1 effectué.")

    # Encodage des variables ordinales
    ordinal_columns = ['City', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Financial Stress']
    enc = LabelEncoder()
    for col in ordinal_columns:
        if col in df.columns:
            df[col] = enc.fit_transform(df[col].astype(str))
            print(f"Encodage de '{col}' effectué.")

    # Encodage des variables binaires
    binary_columns = ['Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            print(f"Encodage de '{col}' en 0 et 1 effectué.")

    # Sauvegarde du dataset prétraité
    df.to_csv(output_file, index=False)
    print(f"Nouveau fichier CSV enregistré sous '{output_file}'.")

except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' est introuvable. Vérifie le chemin du fichier.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")

# Deuxième partie : Transformation et encodage supplémentaires

# Chargement du dataset prétraité
df = pd.read_csv(output_file)
print("Données prétraitées chargées avec succès.")

# Catégorisation de l'âge
if 'Age' in df.columns:
    age_bins = [0, 12, 17, 25, 65, 100]
    age_labels = ['Enfant', 'Adolescence', 'Jeune', 'Adulte', 'Senior']
    df['Age_cat'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    df = pd.get_dummies(df, columns=['Age_cat'], dtype=int)
    print("Catégorisation et encodage One-Hot de l'âge effectués.")
    
    # Réorganisation des colonnes
    colonnes = list(df.columns)
    colonnes_age_cat = [col for col in colonnes if col.startswith('Age_cat_')]
    position_age = colonnes.index('Age') if 'Age' in colonnes else -1
    if position_age != -1:
        colonnes.remove('Age')
        colonnes = colonnes[:position_age] + colonnes_age_cat + colonnes[position_age:]
    df = df[colonnes]
    df.drop(columns=['Age'], inplace=True, errors='ignore')
    print("Colonne 'Age' supprimée après transformation.")
else:
    print("Erreur : La colonne 'Age' est introuvable dans le dataset.")

# Encodage One-Hot de plusieurs colonnes catégoriques
ohe_columns = ['Academic Pressure', 'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 'Study Satisfaction', 'Dietary Habits', 'Financial Stress']
for col in ohe_columns:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], dtype=int)
        print(f"Encodage One-Hot de '{col}' effectué.")

# Sauvegarde du dataset transformé
df.to_csv(output_file, index=False)
print("Les modifications ont été enregistrées avec succès dans le fichier :", output_file)