"sampling": {
    "enabled": false,
    "sampling_type": "gmm or random",
    "fraction": 0.2
  },


#Pretraitement de données de Iris
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


# Supprimer des lignes avec la classe Virginica
df = df[df['species'].isin(['setosa', 'versicolor'])]
df['species'].replace({'versicolor': 0}, inplace=True)
df['species'].replace({'setosa': 1}, inplace=True)

# Définition de la colonne cible
target_column = config["target_column"]
if target_column not in df.columns:
    raise ValueError(f" La colonne cible '{target_column}' est absente des données.")

print(f" Colonne cible définie : {target_column}")



#################################################

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



# Définition de la colonne cible
target_column = "Depression"

# Vérification si la colonne cible existe dans le DataFrame
if target_column not in df.columns:
    raise ValueError(f" La colonne cible '{target_column}' est absente des données.")

print(f" Colonne cible définie : {target_column}")

# Suppression de la colonne 'id' qui est généralement inutile pour l'analyse
df.drop(['id'],axis = 1,inplace = True)

# Encodage de la variable catégorielle 'Gender' en variables binaires
df = pd.get_dummies(df, columns =['Gender'],drop_first = True)

# Encodage des variables catégorielles ordinales en valeurs numériques
enc = LabelEncoder()
df['City'] = enc.fit_transform(df['City'])
df['Profession']= enc.fit_transform(df['Profession'])
df['Sleep Duration'] = enc.fit_transform(df['Sleep Duration'])
df['Dietary Habits'] = enc.fit_transform(df['Dietary Habits'])
df['Degree'] = enc.fit_transform(df['Degree'])
df['Financial Stress'] = enc.fit_transform(df['Financial Stress'])

# Encodage des variables binaires sous forme de variables indicatrices
df = pd.get_dummies(df, columns = ['Family History of Mental Illness'], drop_first = True)
df = pd.get_dummies(df, columns = ['Have you ever had suicidal thoughts ?'], drop_first = True)



#######################################

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
Diagnosis : Diagnostic final (Benign ou Malignant).


# Suppression de la colonne 'patient_id' 
df = df.drop('patient_id', axis=1)

# Vérification des dimensions du DataFrame après suppression de la colonne
df.shape  

# Normalisation des noms de colonnes : remplacement des espaces par des underscores et conversion en minuscules
df.columns = df.columns.str.replace(" ", "_").str.lower()

# Affichage des trois premières lignes du DataFrame
df.head(3)

# Vérification du nombre de valeurs manquantes dans chaque colonne
df.isna().sum()

# Sélection des colonnes de type numérique (int et float)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_columns  # Affichage des colonnes numériques

# Définition des colonnes catégorielles (variables qualitatives)
categorical_cols = ['gender', 'country', 'ethnicity', 'family_history', 'radiation_exposure', 
                    'iodine_deficiency', 'smoking', 'obesity', 'diabetes', 'thyroid_cancer_risk', 
                    'diagnosis']

# Affichage des valeurs uniques pour chaque colonne catégorielle (utile pour vérifier les données avant l'encodage)
for col in categorical_cols:
    print(col, ":", df[col].unique())

# Transformation des valeurs de la colonne 'diagnosis' en format numérique (0 pour Benign, 1 pour Malignant)
df['diagnosis'] = df['diagnosis'].map({'Benign': 0, 'Malignant': 1})

# Transformation des niveaux de risque du cancer thyroïdien en valeurs numériques ordonnées (Low=1, Medium=2, High=3)
df['thyroid_cancer_risk'] = df['thyroid_cancer_risk'].map({'Low': 1, 'Medium': 2, 'High': 3})

# Liste des colonnes catégorielles nominales (sans ordre particulier)
nominal_cols = ['gender', 'country', 'ethnicity', 'family_history', 'radiation_exposure', 
                'iodine_deficiency', 'smoking', 'obesity', 'diabetes']

# Encodage des variables catégorielles nominales en variables binaires (one-hot encoding), suppression de la première catégorie pour éviter la colinéarité
df2 = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)

# Affichage des trois premières lignes du DataFrame après l'encodage
df2.head(3)

# Définition de la colonne cible pour la modélisation
target_column = "diagnosis"

# Vérification de la présence de la colonne cible dans les données
if target_column not in df.columns:
    raise ValueError(f" La colonne cible '{target_column}' est absente des données.")

# Confirmation que la colonne cible est bien définie
print(f" Colonne cible définie : {target_column}")

# Mise à jour du DataFrame avec les transformations effectuées
df = df2

# Affichage des informations générales sur le DataFrame final (types de données, nombre de valeurs non nulles, etc.)
df2.info()



######################################################

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



# Étape 1 : Vérification des valeurs manquantes
missing_data = df.isnull().sum()  # Calcule le nombre de valeurs manquantes pour chaque colonne
print(missing_data)  # Affiche les résultats pour voir quelles colonnes contiennent des valeurs NaN

# Étape 2 : Création d'une colonne binaire pour la qualité du vin
df['is_good'] = (df.quality >= round(df['quality'].mean())).astype(int)  
# Convertit la colonne "quality" en une variable binaire "is_good"
# Si la qualité est supérieure ou égale à la moyenne arrondie, alors is_good = 1 (bon vin), sinon 0 (mauvais vin)

df.drop(['quality'], axis=1, inplace=True)  
# Supprime la colonne "quality" car elle est maintenant remplacée par "is_good"

# Étape 3 : Séparation des variables explicatives (X) et de la variable cible (y)
X = df.drop('is_good', axis=1)  # X contient toutes les colonnes sauf "is_good" (les caractéristiques)
y = df['is_good']  # y est la colonne cible ("is_good") utilisée pour la classification






    
    # Chemin du fichier Excel
excel_path = "resultats_iterations.xlsx"

# Lire l'ancien fichier Excel s'il existe
if os.path.exists(excel_path):
    old_results_df = pd.read_excel(excel_path)
    print("\n📂 Contenu de old_results_df (anciens résultats) :")
    display(old_results_df)  # Affichage interactif
    # print(old_results_df)  # Affichage brut si nécessaire
else:
    old_results_df = pd.DataFrame()
    print("\n⚠️ Aucun ancien résultat trouvé. old_results_df est vide.")

# Afficher les nouveaux résultats
print("\n📊 Contenu de results_df (nouveaux résultats) :")
display(results_df)  # Affichage interactif
# print(results_df)  # Affichage brut si nécessaire



results_df = results_df.iloc[0:0] 
display(results_df)

results_df.drop(results_df.index, inplace=True)




