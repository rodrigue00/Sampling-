import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# Fonction de sampling avec GMM
# Utilisation des paramètres définis dans config.json
def gmm_sampling(df, target_column, config):
    
    """
    Effectue un sampling avec GMM (Gaussian Mixture Model).
    Les échantillons générés sont équilibrés selon les catégories de la cible.

    Paramètres:
    - df (DataFrame): Le DataFrame d'origine.
    - target_column (str): La colonne cible.
    - config (dict): Le fichier de configuration.
    - iteration (int): Numéro de l'itération (utilisé comme random_state).

    Retourne:
    - DataFrame contenant les échantillons générés.
    """
    
    n_components_range = config['sampling']['gmm']['n_components_range']
    covariance_types = config['sampling']['gmm']['covariance_types']
    reduction_rate = config['sampling']['fraction']

    # Séparation des features et de la cible
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Calcul du nombre total de points à générer
    total_samples = int(len(X) * reduction_rate)

    # Fonction pour optimiser le GMM et générer des échantillons
    def generate_samples(category):
        X_cat = X[y == category]
        best_bic, best_gmm = np.inf, None
        for n in n_components_range:
            for cov_type in covariance_types:
                gmm = GaussianMixture(n_components=n, covariance_type=cov_type)
                gmm.fit(X_cat)
                bic = gmm.bic(X_cat)
                if bic < best_bic:
                    best_bic, best_gmm = bic, gmm
        n_samples = total_samples // len(np.unique(y))
        X_generated, _ = best_gmm.sample(n_samples)
        return np.round(X_generated, 1), np.full(n_samples, category)

    # Génération des échantillons pour chaque catégorie
    X_new, y_new = [], []
    for category in np.unique(y):
        X_cat, y_cat = generate_samples(category)
        X_new.append(X_cat)
        y_new.append(y_cat)

    # Concaténation des échantillons générés
    X_new = np.vstack(X_new)
    y_new = np.hstack(y_new)

    # Création du DataFrame final
    column_names = list(X.columns) + [target_column]
    df_new = pd.DataFrame(X_new, columns=column_names[:-1])
    df_new[target_column] = y_new

    return df_new