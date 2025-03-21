import json
import os

# 📌 Définition du fichier log
log_file = "results_log.txt"

def collect_results(iteration, config, file_path, target_column, sampling_status, sampling_time, 
                     train_percent, num_train_sampled, test_percent, num_test_sampled, 
                     model_name, model, model_params, train_time_before, 
                     accuracy_before, f1_before, train_time_after, best_params, 
                     accuracy_after, f1_after):
    """
    Collecte toutes les informations d'une itération sous forme de dictionnaire et les enregistre.
    """

    iteration_results = {
        "Itération": iteration,
        "Dataset utilisé": file_path,
        "Colonne cible": target_column,
        "Échantillonnage": sampling_status,
        "Tps Sampling": sampling_time,
        "Entrain %": train_percent,
        "Nbre Entrain": num_train_sampled,
        "Test %": test_percent,
        "Nbre Test": num_test_sampled,
    }

    # 📌 Ajout des informations spécifiques au sampling GMM
    if config["sampling"]["gmm"]["enabled"]:
        iteration_results["Sampling GMM"] = {
            "n_components_range": config["sampling"]["gmm"]["n_components_range"],
            "covariance_types": config["sampling"]["gmm"]["covariance_types"],
            "reduction_rate": config["sampling"]["gmm"].get("reduction_rate", "Non spécifié"),
            "best_bic": globals().get("best_bic", "Non spécifié"),
            "best_gmm": str(globals().get("best_gmm", "Non spécifié")),
            "n_samples_generated": globals().get("n_samples", "Non spécifié"),
        }

    # 📌 Ajout des informations spécifiques au sampling DDBS
    elif config["sampling"]["ddbs"]["enabled"]:
        iteration_results["Sampling DDBS"] = {
            "distance_metric": config["sampling"]["ddbs"]["distance_metric"],
            "distribution_type": config["sampling"]["ddbs"]["distribution_type"],
            "percentage_per_class": config["sampling"]["ddbs"]["percentage_per_class"],
            "total_selected_samples": num_train_sampled
        }

    # 📌 Ajout des informations spécifiques au Random Sampling
    elif config["sampling"]["random"]["enabled"]:
        iteration_results["Sampling Random"] = {
            "fraction": config["sampling"]["fraction"],
            "total_selected_samples": num_train_sampled
        }

    # 📌 Ajout des informations du modèle et des performances
    iteration_results["Modèle"] = model_name
    iteration_results["Paramètres initiaux"] = {
        param: value for param, value in model.get_params().items() if param in model_params
    }
    iteration_results["Tps Entrain Avant"] = train_time_before
    iteration_results["Accuracy Avant"] = accuracy_before
    iteration_results["F1-score Avant"] = f1_before
    iteration_results["Tps Entrain Après"] = train_time_after
    iteration_results["Paramètres optimisés"] = best_params
    iteration_results["Accuracy Après"] = accuracy_after
    iteration_results["F1-score Après"] = f1_after

    return iteration_results  # Retourne le dictionnaire contenant les résultats
