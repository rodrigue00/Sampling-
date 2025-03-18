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

try:
    # Chargement du dataset
    df = pd.read_csv(file_path)
    print("Données chargées avec succès.")

    # Définition de la colonne cible
    target_column = "Depression"

    # Vérification si la colonne cible existe
    if target_column not in df.columns:
        raise ValueError(f"La colonne cible '{target_column}' est absente des données.")
    print(f"Colonne cible définie : {target_column}")

    # Suppression de la colonne 'id' si elle existe
    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)
        print("Colonne 'id' supprimée.")

    # Encodage de la variable 'Gender' en 0 et 1
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        print("Encodage de 'Gender' en 0 et 1 effectué.")

    # Encodage des variables ordinales avec LabelEncoder
    enc = LabelEncoder()
    ordinal_columns = ['City', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Financial Stress']
    
    for col in ordinal_columns:
        if col in df.columns:
            df[col] = enc.fit_transform(df[col].astype(str))  # Conversion en str pour éviter les erreurs
            print(f"Encodage de '{col}' effectué.")

    # Encodage des variables binaires en 0 et 1
    binary_columns = ['Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']
    
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            print(f"Encodage de '{col}' en 0 et 1 effectué.")

    # Définition du chemin de sortie
    output_file = 'data/preprocessed_student_depression_dataset.csv'
    
    # Sauvegarde du dataset prétraité
    df.to_csv(output_file, index=False)
    print(f"\nNouveau fichier CSV enregistré sous '{output_file}'.")

    # Affichage des premières lignes du dataset prétraité
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Dataset Prétraité", dataframe=df)

except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' est introuvable. Vérifie le chemin du fichier.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")














import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Définition du chemin du fichier
file_path = 'data/preprocessed_student_depression_dataset.csv'

# Chargement du dataset
df = pd.read_csv(file_path)
print("Données chargées avec succès.")

# Vérifier si la colonne "Age" existe
if 'Age' in df.columns:
    # Définir les nouvelles catégories d'âge avec des intervalles plus précis
    bins = [0, 12, 17, 25, 65, 100]  # Définition des intervalles
    labels = ['Enfant', 'Adolescence', 'Jeune', 'Adulte', 'Senior']  # Noms des catégories

    # Appliquer la catégorisation sur la colonne Age
    df['Age_cat'] = pd.cut(df['Age'], bins=bins, labels=labels)

    # Appliquer One-Hot Encoding sur la colonne 'Age_cat' (nom corrigé)
    df_encoded = pd.get_dummies(df, columns=['Age_cat']).astype(int)

    # Afficher les données mises à jour
    from IPython.display import display
    display(df_encoded)
else:
    print("Erreur : La colonne 'Age' est introuvable dans le dataset.")


# réordonne 
# Liste de toutes les colonnes du DataFrame
colonnes = list(df_encoded.columns)

# Identifier les colonnes créées par One-Hot Encoding pour Age_cat
colonnes_age_cat = [col for col in colonnes if col.startswith('Age_cat_')]

# Supprimer ces colonnes de la liste initiale
for col in colonnes_age_cat:
    colonnes.remove(col)

# Trouver la position de la colonne 'Age'
position_age = colonnes.index('Age')

# Insérer les colonnes encodées juste après 'Age'
for i, col in enumerate(colonnes_age_cat):
    colonnes.insert(position_age + 1 + i, col)

# Réordonner le DataFrame selon la nouvelle liste de colonnes
df_reorganise = df_encoded[colonnes]

# Afficher le DataFrame réorganisé
from IPython.display import display
display(df_reorganise)
