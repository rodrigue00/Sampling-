import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def compute_distance(X, Y, metric="manhattan"):
    """Calcule la distance entre les matrices X et Y selon la métrique choisie."""
    if metric == "manhattan":
        return cdist(X, Y, metric='cityblock')  # Distance de Manhattan
    elif metric == "euclidean":
        return cdist(X, Y, metric='euclidean')  # Distance Euclidienne
    else:
        raise ValueError("Métrique non supportée. Choisissez 'manhattan' ou 'euclidean'.")

def diversified_distance_based_sampling(df, target_column, config):
    """
    Applique Diversified Distance-Based Sampling (DDBS) avec la configuration fournie.

    Arguments :
    - df : DataFrame contenant les données.
    - target_column : Nom de la colonne cible.
    - config : Configuration JSON.

    Retourne :
    - DataFrame échantillonné
    """
    percentage_per_class = config["sampling"]["fraction"]
    distance_metric = config["sampling"]["ddbs"]["distance_metric"]
    distribution_type = config["sampling"]["ddbs"]["distribution_type"]

    np.random.seed()  # Fixer la graine pour reproductibilité

    features = df.drop(columns=[target_column]).values
    labels = df[target_column].values

    sampled_indices = []
    candidate_indices = list(range(len(df)))

    unique_classes, class_counts = np.unique(labels, return_counts=True)
    samples_per_class = {cls: max(1, int(percentage_per_class * count)) for cls, count in zip(unique_classes, class_counts)}
    class_selected_counts = {cls: 0 for cls in unique_classes}

    for cls in unique_classes:
        cls_indices = [idx for idx in candidate_indices if labels[idx] == cls]
        first_sample = np.random.choice(cls_indices)
        sampled_indices.append(first_sample)
        candidate_indices.remove(first_sample)
        class_selected_counts[cls] += 1

    while any(class_selected_counts[cls] < samples_per_class[cls] for cls in unique_classes):
        remaining_classes = [cls for cls in unique_classes if class_selected_counts[cls] < samples_per_class[cls]]

        if not remaining_classes:
            break

        sampled_features = features[sampled_indices]
        candidate_features = features[candidate_indices]
        distances = compute_distance(sampled_features, candidate_features, metric=distance_metric)

        max_distance = distances.max()

        if distribution_type == "uniform":
            selected_distance = np.random.randint(1, max_distance + 1)
        elif distribution_type == "binomial":
            selected_distance = np.random.binomial(n=max_distance, p=0.5)
        elif distribution_type == "geometric":
            selected_distance = min(max_distance, np.random.geometric(0.3))
        else:
            raise ValueError("Distribution non supportée. Choisissez 'uniform', 'binomial', ou 'geometric'.")

        best_candidate = None
        best_candidate_class = None
        min_distances = distances.min(axis=0)

        for idx, candidate_idx in enumerate(candidate_indices):
            candidate_class = labels[candidate_idx]
            if candidate_class in remaining_classes:
                if min_distances[idx] >= selected_distance:
                    best_candidate = candidate_idx
                    best_candidate_class = candidate_class
                    break

        if best_candidate is None:
            remaining_indices = [i for i, idx in enumerate(candidate_indices) if labels[idx] in remaining_classes]
            if remaining_indices:
                best_candidate_idx = remaining_indices[np.argmax(np.take(min_distances, remaining_indices))]
                best_candidate = candidate_indices[best_candidate_idx]
                best_candidate_class = labels[best_candidate]

        sampled_indices.append(best_candidate)
        candidate_indices.remove(best_candidate)
        class_selected_counts[best_candidate_class] += 1

    return df.iloc[sampled_indices]
