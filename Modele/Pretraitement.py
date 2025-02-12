# 🔹 Gérer les valeurs manquantes
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Chargement du DataFrame (Remplacez ceci par votre propre méthode de chargement)
# df = pd.read_csv("data/raw_data.csv")

# Vérifier les valeurs manquantes
print("\nValeurs manquantes avant traitement :")
print(df.isnull().sum())

# Remplir les valeurs manquantes pour les variables numériques avec la médiane
if not df.empty:
    df.fillna(df.median(numeric_only=True), inplace=True)

# Remplir les valeurs manquantes pour les variables catégoriques avec le mode
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:  # Vérifier s'il y a des valeurs manquantes
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nValeurs manquantes traitées.")

# Gestion des doublons
nb_duplicated = df.duplicated().sum()
if nb_duplicated > 0:
    df.drop_duplicates(inplace=True)
    print(f"\n{nb_duplicated} doublon(s) supprimé(s).")
else:
    print("\nAucun doublon détecté.")

# Encodage des variables catégoriques
categorical_cols = df.select_dtypes(include=['object']).columns

if len(categorical_cols) > 0:
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))  # Conversion en string pour éviter les erreurs
    print("\nEncodage des variables catégoriques effectué.")

# Normalisation des variables numériques
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

if len(numerical_cols) > 0:
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("\nNormalisation des données terminée.")

# Affichage des données après prétraitement
print("\nAperçu des données après prétraitement :")
print(df.head())

# Sauvegarde des données prétraitées
output_path = "data/cleaned_data.csv"
df.to_csv(output_path, index=False)
print(f"\nDonnées prétraitées enregistrées sous '{output_path}'.")











 #  Gestion du déséquilibre des classes
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("\n Distribution des classes avant équilibrage :", Counter(y))

    # Choisir entre sur-échantillonnage ou sous-échantillonnage
    strategy = "smote"  # Options : "smote", "undersampling", "none"

    if strategy == "smote":
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print("\n Sur-échantillonnage avec SMOTE effectué.")
    elif strategy == "undersampling":
        undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        print("\n Sous-échantillonnage effectué.")
    else:
        X_resampled, y_resampled = X, y

    print("\n Répartition des classes après équilibrage :", Counter(y_resampled))

    #  Sauvegarde des données prétraitées
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_col] = y_resampled
    output_path = "data/cleaned_balanced_data.csv"
    df_resampled.to_csv(output_path, index=False)

    print(f"\n Données équilibrées enregistrées sous '{output_path}'.")