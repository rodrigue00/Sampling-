import pandas as pd
import json
import optuna
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# Charger la configuration depuis config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

file_path = config["file_path"]
random_state = config.get("random_state", 42)
model_save_path = config["model_save_path"]

# Charger le dataset CSV avec gestion des erreurs
df = pd.read_csv(file_path)
print(" Données chargées avec succès.")


# Définition de la colonne cible
target_column = ["target_column"]
if target_column not in df.columns:
    raise ValueError(f" La colonne cible '{target_column}' est absente des données.")

print(f" Colonne cible définie : {target_column}")

# Initialisation du DataFrame pour stocker les résultats
columns = [
    "Itération", "Échantillonnage", "Modèle", "Paramètres Initiaux",
    "Accuracy Avant", "F1-score Avant", "Paramètres Optimisés",
    "Accuracy Après", "F1-score Après"
]
results_df = pd.DataFrame(columns=columns)

# Exécuter 30 itérations
for iteration in tqdm(range(1, 31), desc=" Itérations en cours"):
    print(f"\n Exécution de l'itération {iteration}...\n")

    # Appliquer l'échantillonnage si activé
    if config["sampling"]["enabled"]:
        fraction = config["sampling"]["fraction"]
        df_sampled = df.sample(frac=fraction, random_state=random_state)
        sampling_status = "Oui"
        print(f" Échantillonnage activé : {fraction*100:.1f}% des données utilisées.")
    else:
        df_sampled = df
        sampling_status = "Non"
        print(" Aucun échantillonnage appliqué, utilisation des données complètes.")

    # Séparation des données en train/test
    test_size = config["train_test_split"]["test_size"]
    X = df_sampled.drop(columns=[target_column])
    y = df_sampled[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"\n Données séparées : {100 * (1 - test_size):.0f}% entraînement, {100 * test_size:.0f}% test.")

    # Initialiser les modèles
    models = {}
    for model_name, model_params in config["models"].items():
        if model_params["enabled"]:
            if model_name == "Decision Tree":
                models[model_name] = DecisionTreeClassifier(max_depth=model_params["max_depth"], random_state=random_state)
            elif model_name == "Random Forest":
                models[model_name] = RandomForestClassifier(
                    n_estimators=model_params["n_estimators"],
                    max_depth=model_params["max_depth"],
                    random_state=random_state
                )
            elif model_name == "SVM":
                models[model_name] = SVC(C=model_params["C"], kernel=model_params["kernel"])
            elif model_name == "Neural Network":
                models[model_name] = MLPClassifier(
                    hidden_layer_sizes=tuple(model_params["hidden_layer_sizes"]),
                    learning_rate_init=model_params["learning_rate_init"],
                    max_iter=500,
                    random_state=random_state
                )
            elif model_name == "XGBoost":
                models[model_name] = XGBClassifier(
                    n_estimators=model_params["n_estimators"],
                    max_depth=model_params["max_depth"],
                    learning_rate=model_params["learning_rate"],
                    random_state=random_state    
                )

    # Entraînement et évaluation des modèles
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_before = accuracy_score(y_test, y_pred)
        f1_before = f1_score(y_test, y_pred, average="weighted")
        
        print(f"\n📊 Résultats du modèle '{model_name}' avant optimisation :")
        print(f"   - Accuracy: {accuracy_before:.4f}")
        print(f"   - F1-score: {f1_before:.4f}")

        best_params = {}
        best_model = model
        accuracy_after, f1_after = "Non optimisé", "Non optimisé"

        # Optimisation avec Optuna
        if config["use_optuna"]:
            def objective(trial):
                if model_name == "Decision Tree":
                    max_depth = trial.suggest_int("max_depth", 2, 20)
                    model_opt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
                elif model_name == "Random Forest":
                    n_estimators = trial.suggest_int("n_estimators", 10, 200)
                    max_depth = trial.suggest_int("max_depth", 2, 20)
                    model_opt = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                elif model_name == "SVM":
                    C = trial.suggest_loguniform("C", 0.1, 10)
                    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
                    model_opt = SVC(C=C, kernel=kernel)
                elif model_name == "Neural Network":
                    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50)])
                    learning_rate_init = trial.suggest_loguniform("learning_rate_init", 0.0001, 0.1)
                    model_opt = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, max_iter=500, random_state=random_state)
                elif model_name == "XGBoost":
                    n_estimators = trial.suggest_int("n_estimators", 50, 500)
                    max_depth = trial.suggest_int("max_depth", 2, 20)
                    learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.3)
                    model_opt = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)

                model_opt.fit(X_train, y_train)
                y_pred_opt = model_opt.predict(X_test)
                return accuracy_score(y_test, y_pred_opt)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=config["n_trials"])

            best_params = study.best_trial.params
            print(f"\n Meilleurs paramètres trouvés pour '{model_name}' : {best_params}")

            if config["retrain_with_best_params"]:
                best_model = model.__class__(**best_params, random_state=random_state)
                best_model.fit(X_train, y_train)
                y_pred_opt = best_model.predict(X_test)
                accuracy_after = accuracy_score(y_test, y_pred_opt)
                f1_after = f1_score(y_test, y_pred_opt, average="weighted")
                
            print(f"\n Modèle après optimisation ({model_name}):")
            print(f"   - Accuracy: {accuracy_after}")
            print(f"   - F1-score: {f1_after}")

            # Sauvegarde du meilleur modèle
            if config["save_best_model"]:
                model_filename = f"best_model_{model_name}.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(best_model, f)

        results_df = pd.concat([results_df, pd.DataFrame([{
            "Itération": iteration,
            "Échantillonnage": sampling_status,
            "Modèle": model_name,
            "Paramètres Initiaux": {param: value for param, value in model.get_params().items() if param in model_params},
            "Accuracy Avant": accuracy_before,
            "F1-score Avant": f1_before,
            "Paramètres Optimisés": best_params,
            "Accuracy Après": accuracy_after,
            "F1-score Après": f1_after
        }])], ignore_index=True)

# Sauvegarde des résultats
# Définition du chemin du fichier Excel
excel_path = "resultats_iterations.xlsx"

# Vérifier si le fichier existe
if os.path.exists(excel_path):
    # Ajouter les nouveaux résultats sans charger les anciens
    with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay') as writer:
        results_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
else:
    # Si le fichier n'existe pas, on crée un nouveau fichier avec les nouveaux résultats
    results_df.to_excel(excel_path, index=False)

print("\n Les nouveaux résultats ont été ajoutés dans 'resultats_iterations.xlsx'.")

# Réinitialisation du DataFrame après la sauvegarde (conserve les colonnes, supprime les lignes)
results_df.drop(results_df.index, inplace=True)