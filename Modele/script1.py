import numpy as np
import pandas as pd
import json
import optuna
import pickle
import os
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# Import collection des resultat
from resultat.collect_results import collect_results

# Import sécurisé de gmm_sampling
try:
    from sampling.gmm_sampling import gmm_sampling
except ModuleNotFoundError:
    raise ImportError(" Impossible d'importer gmm_sampling. Vérifiez le chemin du fichier.")

# Import sécurisé de ddbs_sampling
try:
    from sampling.ddbs_sampling import diversified_distance_based_sampling
except ModuleNotFoundError:
    raise ImportError("Impossible d'importer ddbs_sampling. Vérifiez le chemin du fichier.")


# Charger la configuration depuis config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

file_path = config["file_path"]
model_save_path = config["model_save_path"]
num_iterations = config.get("num_iterations", 2)  #  Nombre d'itérations

# Charger le dataset CSV avec gestion des erreurs
df = pd.read_csv(file_path)
print(" Données chargées avec succès.")

# Définition de la colonne cible
target_column = config["target_column"]
if target_column not in df.columns:
    raise ValueError(f" La colonne cible '{target_column}' est absente des données.")

print(f" Colonne cible définie : {target_column}")

# Initialisation du DataFrame pour stocker les résultats
columns = [
    "Itération", "Échantillonnage","Tps Sampling","Entrain %", "Nbre Entrain", "Test %", "Nbre Test", "Modèle", "Paramètres Initiaux",
    "Tps Entrain Avant", "Accuracy Avant","F1-score Avant", "Tps Entrain Après", "Paramètres Optimisés", "Accuracy Après", "F1-score Après"    
]
results_df = pd.DataFrame(columns=columns)



#  Définition du fichier log pour les résultats
log_file = "results_log.txt"

def save_results(iteration_results):
    """
    Ajoute les résultats d'une itération dans `results_log.txt`
    sans recréer le fichier à chaque fois.
    """
    try:
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("Log des résultats\n\n")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(iteration_results, indent=4, ensure_ascii=False) + "\n\n")

        print(f"Résultats enregistrés dans {log_file}")

    except Exception as e:
        print(f" Erreur lors de l'enregistrement des résultats : {e}")



# Exécuter 30 itérations

for iteration in tqdm(range(1, num_iterations  + 1), desc=" Itérations en cours"):
    print(f"\n Exécution de l'itération {iteration}...\n")

     # Séparation des données en Train/Test avant sampling
    test_size = config["train_test_split"]["test_size"]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # CONSERVER les données SANS échantillonnage pour les comparer plus tard
    X_train_no_sampling, y_train_no_sampling = X_train.copy(), y_train.copy()
    
    #  Appliquer le sampling UNIQUEMENT sur les données d'entraînement
    start_sampling_time = time.time()
    
    # Appliquer le sampling si activé
    if config["sampling"]["enabled"]:
        # Si le sampling type est GMM, appliquer le sampling GMM
        if config["sampling"]["gmm"]["enabled"]:
            df_train_sampled = gmm_sampling(pd.concat([X_train, y_train], axis=1), target_column, config)
            X_train_sampled = df_train_sampled.drop(columns=[target_column])
            y_train_sampled = df_train_sampled[target_column]
            sampling_status = "Oui (GMM)"
            print("Échantillonnage GMM appliqué")
        
        # Si le sampling type est Random, utiliser Pandas sample()
        elif config["sampling"]["random"]["enabled"]:
            fraction = config["sampling"]["fraction"]
            sampled_indices = X_train.sample(frac=fraction).index
            X_train_sampled, y_train_sampled = X_train.loc[sampled_indices], y_train.loc[sampled_indices]
            sampling_status = "Oui (Random)"
            print(f" Échantillonnage aléatoire activé : {fraction*100:.1f}% des données utilisées.")
        
        # Si le sampling type est ddbs, utiliser Pandas sample()    
        elif config["sampling"]["ddbs"]["enabled"]:
            df_train_sampled = diversified_distance_based_sampling(pd.concat([X_train, y_train], axis=1), target_column, config)
            X_train_sampled = df_train_sampled.drop(columns=[target_column])
            y_train_sampled = df_train_sampled[target_column]
            sampling_status = f"Oui (DDBS - {config['sampling']['ddbs']['distance_metric']}, {config['sampling']['ddbs']['distribution_type']})"
            print("Échantillonnage DDBS appliqué.")
        
        # Si le type est inconnu, lever une erreur
        else:
            raise ValueError(f" Type d'échantillonnage inconnu : '{sampling_type}'")

    else:
        # Si le sampling est désactivé, utiliser le dataset original
        X_train_sampled, y_train_sampled = X_train, y_train
        sampling_status = "Non"
        print(" Aucun échantillonnage appliqué, utilisation des données complètes.")



   # Sélectionner les bonnes données d'entraînement (avec ou sans échantillonnage)
    X_train_used, y_train_used = (X_train_sampled, y_train_sampled) if config["sampling"]["enabled"] else (X_train, y_train)
    

    # Sélectionner les bonnes données d'entraînement 
    X_train_sampling, y_train_sampling = X_train_sampled, y_train_sampled  # AVEC sampling


    # Calcul du temps pris pour le sampling
    sampling_time = round(time.time() - start_sampling_time , 2) # Calcul du temps pris par le sampling
    print(f"\n Temps samping  : {sampling_time}.")
    
    #  Mise à jour du nombre d'échantillons après sampling
    num_train_used  = len(X_train_used)
    num_test_used  = len(X_test) # Le test set reste inchangé

    num_train_no_sampling = len(X_train_no_sampling)  # Taille des données sans échantillonnage


    print(f"\nDonnées après échantillonnage (Train) : {num_train_used } échantillons.")
    print(f"Données de test non modifiées : {num_test_used } échantillons.")
    
    print(f" Données sans échantillonnage (Train) : {num_train_no_sampling} échantillons.")
    
    # Calcul du pourcentage des données après échantillonnage
    train_percent = (num_train_used / len(df)) * 100
    test_percent = (num_test_used / len(df)) * 100

    train_percent_no_sampling = (num_train_no_sampling / len(df)) * 100

    
    print(f"\n Pourcentage Entraînement : {train_percent:.2f}%")
    print(f" Pourcentage Test : {test_percent:.2f}%")

    print(f" Pourcentage Entraînement (Sans Sampling) : {train_percent_no_sampling:.2f}%")



    # Initialiser les modèles


    models_no_sampling = {}  # Modèles entraînés SANS échantillonnage
    models_sampling = {}  # Modèles entraînés AVEC échantillonnage
    
    models = {}
    for model_name, model_params in config["models"].items():
        if model_params["enabled"]:
            if model_name == "Decision Tree":
                models[model_name] = DecisionTreeClassifier(max_depth=model_params["max_depth"])
            elif model_name == "Random Forest":
                models[model_name] = RandomForestClassifier(
                    n_estimators=model_params["n_estimators"],
                    max_depth=model_params["max_depth"]
                )
            elif model_name == "SVM":
                models[model_name] = SVC(C=model_params["C"], kernel=model_params["kernel"])
            elif model_name == "Neural Network":
                models[model_name] = MLPClassifier(
                    hidden_layer_sizes=tuple(model_params["hidden_layer_sizes"]),
                    learning_rate_init=model_params["learning_rate_init"],
                    max_iter=500
                )
            elif model_name == "XGBoost":
                models[model_name] = XGBClassifier(
                    n_estimators=model_params["n_estimators"],
                    max_depth=model_params["max_depth"],
                    learning_rate=model_params["learning_rate"]    
                )

    # Entraînement et évaluation des modèles
    for model_name, model in models.items():

        # Entraînement du modèle avec les bonnes données
        start_train_time = time.time()  # Début du chronométrage de l'entraînement avant optimisation
        model.fit(X_train_used, y_train_used)
        train_time_before = round(time.time() - start_train_time , 2)  # Temps pris pour l'entraînement avant optimisation

        # Entraînement du modèle SANS échantillonnage
        model.fit(X_train_no_sampling, y_train_no_sampling)
        y_pred_no_sampling = model.predict(X_test)
        

        #  Entraînement du modèle AVEC échantillonnage
        model.fit(( X_train_sampling, y_train_sampling)
        y_pred_sampling = model.predict(X_test)


        #  Comparaison des scores avant optimisation
        accuracy_no_sampling = accuracy_score(y_test, y_pred_no_sampling)
        f1_no_sampling = f1_score(y_test, y_pred_no_sampling, average="weighted")

        accuracy_sampling = accuracy_score(y_test, y_pred_sampling)
        f1_sampling = f1_score(y_test, y_pred_sampling, average="weighted")



        print(f"\n 📊 Comparaison des résultats avant optimisation pour '{model_name}':")
        print(f"   - Accuracy (Sans Sampling) : {accuracy_no_sampling:.4f}")
        print(f"   - Accuracy (Avec Sampling) : {accuracy_sampling:.4f}")
        print(f"   - F1-score (Sans Sampling) : {f1_no_sampling:.4f}")
        print(f"   - F1-score (Avec Sampling) : {f1_sampling:.4f}")

        similarity_percentage = np.mean(y_pred_sampling == y_pred_no_sampling) * 100

        print(f"   - Similarité des prédictions entre modèles : {similarity_percentage:.2f}%")
        

        #  Calcul des scores sur TRAIN
        y_train_pred = model.predict(X_train_used)
        train_accuracy = accuracy_score(y_train_used, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average="weighted")

        #  Calcul des scores sur TEST
        y_pred = model.predict(X_test)
        accuracy_before = accuracy_score(y_test, y_pred)
        f1_before = f1_score(y_test, y_pred, average="weighted")
        
        print(f"\n Résultats du modèle '{model_name}' avant optimisation :")
        print(f"   - Accuracy Train: {train_accuracy:.4f}")
        print(f"   - Accuracy Test: {accuracy_before:.4f}")
        print(f"   - F1-score Train: {train_f1:.4f}")
        print(f"   - F1-score Test: {f1_before:.4f}")

        #  Vérification de l'overfitting / underfitting
        if train_accuracy > accuracy_before + 0.10:
            print(" Overfitting détecté !")
        elif train_accuracy < 0.7 and accuracy_before < 0.7:
            print(" Underfitting détecté !")
        else:
            print(" Bon équilibre entre biais et variance.")

        best_params = {}
        best_model = model
        accuracy_after, f1_after = "Non optimisé", "Non optimisé"

        # Optimisation avec Optuna
        if config["use_optuna"]:
            def objective(trial):
                if model_name == "Decision Tree":
                    max_depth = trial.suggest_int("max_depth", 2, 20)
                    model_opt = DecisionTreeClassifier(max_depth=max_depth)
                elif model_name == "Random Forest":
                    n_estimators = trial.suggest_int("n_estimators", 10, 200)
                    max_depth = trial.suggest_int("max_depth", 2, 20)
                    model_opt = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                elif model_name == "SVM":
                    C = trial.suggest_loguniform("C", 0.1, 10)
                    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
                    model_opt = SVC(C=C, kernel=kernel)
                elif model_name == "Neural Network":
                    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50)])
                    learning_rate_init = trial.suggest_loguniform("learning_rate_init", 0.0001, 0.1)
                    model_opt = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, max_iter=500)
                elif model_name == "XGBoost":
                    n_estimators = trial.suggest_int("n_estimators", 50, 500)
                    max_depth = trial.suggest_int("max_depth", 2, 20)
                    learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.3)
                    model_opt = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

                
                model_opt.fit((X_train_no_sampling, y_train_no_sampling)
                model_opt.fit( X_train_sampling, y_train_sampling)
                 model_opt.fit(X_train_used, y_train_used)
                y_pred_opt = model_opt.predict(X_test)
                return accuracy_score(y_test, y_pred_opt)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=config["n_trials"])

            best_params = study.best_trial.params
            print(f"\n Meilleurs paramètres trouvés pour '{model_name}' : {best_params}")

            # Réentraînement avec les meilleurs paramètres
            if config["retrain_with_best_params"]:
                start_train_time_after = time.time()  # Début du chronométrage de l'entraînement après optimisation
                best_model = model.__class__(**best_params)
                best_model.fit(X_train_used, y_train_used)
                train_time_after = round(time.time() - start_train_time_after, 2) # Temps pris pour l'entraînement après optimisation


                
                # Entraînement du modèle SANS échantillonnage
                best_model.fit(X_train_no_sampling, y_train_no_sampling)
                y_pred_no_sampling_opt =  best_model.predict(X_test)
                
        
                #  Entraînement du modèle AVEC échantillonnage
                best_model.fit(( X_train_sampling, y_train_sampling)
                y_pred_sampling_opt =  best_model.predict(X_test)
        
        
                #  Comparaison des scores avant optimisation
                accuracy_no_sampling_opt = accuracy_score(y_test, y_pred_no_sampling)
                f1_no_sampling = f1_score(y_test, y_pred_no_sampling, average="weighted")
        
                accuracy_sampling_opt = accuracy_score(y_test, y_pred_sampling)
                f1_sampling = f1_score(y_test, y_pred_sampling, average="weighted")
        
        
        
                print(f"\n  Comparaison des résultats avant optimisation pour '{model_name}':")
                print(f"   - Accuracy (Sans Sampling) : {accuracy_no_sampling_opt:.4f}")
                print(f"   - Accuracy (Avec Sampling) : {accuracy_sampling_opt:.4f}")
                print(f"   - F1-score (Sans Sampling) : {f1_no_sampling_opt:.4f}")
                print(f"   - F1-score (Avec Sampling) : {f1_sampling_opt:.4f}")
        
                similarity_percentage_opt = np.mean(y_pred_sampling_opt == y_pred_no_sampling_opt) * 100
        
                print(f"   - Similarité des prédictions entre modèles : {similarity_percentage_opt:.2f}%")
                

                #  Calcul des scores sur TRAIN apres optimisation 
                y_train_pred_opt = best_model.predict(X_train_used)
                train_accuracy_after = accuracy_score(y_train_used, y_train_pred_opt)
                train_f1_after = f1_score(y_train, y_train_pred_opt, average="weighted")
                
                #  Calcul des scores sur TEST apres optimisation 
                y_pred_opt = best_model.predict(X_test)
                accuracy_after = accuracy_score(y_test, y_pred_opt)
                f1_after = f1_score(y_test, y_pred_opt, average="weighted")

            print(f"\n Modèle après optimisation ({model_name}):")
            print(f"   - Accuracy Train: {train_accuracy_after:.4f}")
            print(f"   - Accuracy Test: {accuracy_after:.4f}")
            print(f"   - F1-score Train: {train_f1_after:.4f}")
            print(f"   - F1-score Test: {f1_after:.4f}")

            #  Vérification de l'overfitting / underfitting après optimisation
            if train_accuracy_after > accuracy_after + 0.10:
                print(" Overfitting détecté après optimisation !")
            elif train_accuracy_after < 0.7 and accuracy_after < 0.7:
                print(" Underfitting détecté après optimisation !")
            else:
                print(" Bon équilibre entre biais et variance après optimisation.")
                               

            # Sauvegarde du meilleur modèle
            if config["save_best_model"]:
                model_filename = f"best_model_{model_name}.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(best_model, f)
                print(f" Modèle '{model_name}' sauvegardé sous '{model_filename}'.")

        # Stockage des résultats
        results_df = pd.concat([results_df, pd.DataFrame([{
            "Itération": iteration,
            "Échantillonnage": sampling_status,
            "Tps Sampling": sampling_time,
            "Entrain %": train_percent,
            "Nbre Entrain": num_train_used,
            "Test %": test_percent,
            "Nbre Test": num_test_used,
            "Modèle": model_name,
            "Paramètres Initiaux": {param: value for param, value in model.get_params().items() if param in model_params},
            "Tps Entrain Avant": train_time_before,
            "Accuracy Avant": accuracy_before,
            "F1-score Avant": f1_before,
            "Tps Entrain Après": train_time_after,
            "Paramètres Optimisés": best_params,
            "Accuracy Après": accuracy_after,
            "F1-score Après": f1_after
        }])], ignore_index=True)
        
        

        # 📌 Vérification si on est bien entré dans la collecte des résultats
        print(f" [DEBUG] Début de la collecte des résultats pour l'itération {iteration}...")
        
        #  Enregistrement des résultats dans results_log.txt avec collect_results
        iteration_results =collect_results(
            iteration, config, file_path, target_column, sampling_status, sampling_time,
            train_percent, num_train_sampled, test_percent, num_test_sampled,
            model_name, model, model.get_params(), train_time_before,
            accuracy_before, f1_before, train_time_after, best_params,
            accuracy_after, f1_after
        )

        
        #  Vérification si collect_results() a retourné un dictionnaire valide
        print(f" [DEBUG] Résultats collectés pour l'itération {iteration}:")
        print(json.dumps(iteration_results, indent=4, ensure_ascii=False))
        
        #  Enregistrement des résultats
        print(f" [DEBUG] Enregistrement des résultats dans results_log.txt...")
        save_results(iteration_results)
        print(f" [DEBUG] Résultats enregistrés avec succès !")



# Sauvegarde des résultats
# Définition du chemin du fichier Excel
excel_path = f"resultats_{model_name}_{sampling_status}_{os.path.basename(file_path).split('.')[0]}.xlsx"

# Vérifier si le fichier existe
if os.path.exists(excel_path):
    # Ajouter les nouveaux résultats sans charger les anciens
    with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay') as writer:
        results_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
else:
    # Si le fichier n'existe pas, on crée un nouveau fichier avec les nouveaux résultats
    results_df.to_excel(excel_path, index=False)

print("\n Les nouveaux résultats sont disponible")

# Réinitialisation du DataFrame après la sauvegarde (conserve les colonnes, supprime les lignes)
results_df.drop(results_df.index, inplace=True)

