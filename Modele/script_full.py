import pandas as pd
import json
import pickle
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Charger la configuration depuis le fichier config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

file_path = config["file_path"]
file_type = config["file_type"].lower()

# 2. Charger les données en fonction du type de fichier
try:
    if file_type == "csv":
        df = pd.read_csv(file_path)  # Charger un fichier CSV
    elif file_type == "json":
        df = pd.read_json(file_path)  # Charger un fichier JSON
    elif file_type == "excel":
        df = pd.read_excel(file_path)  # Charger un fichier Excel
    else:
        raise ValueError("Format de fichier non pris en charge : choisissez 'csv', 'json' ou 'excel'.")
    
    # Afficher les premières lignes des données chargées
    print("Données chargées avec succès :")
    
     # 3. Comprendre les données
     
     # Afficher les premières lignes
    print("\n🔍 Aperçu des données :")
    print(df.head())
    
    # Informations générales sur les données
    df.info()

    # Vérifier la taille du dataset
    print("\n Dimensions du dataset (lignes, colonnes) :", df.shape)

    # Vérifier les types de données
    print("\n Types de données par colonne :")
    print(df.dtypes)

    # Vérifier les valeurs manquantes
    print("\n Valeurs manquantes par colonne :")
    print(df.isnull().sum())
    
    # Vérifier le nombre de valeurs uniques par colonne
    print("\n Nombre de valeurs uniques par colonne :")
    print(df.nunique())
    
    # Détecter les doublons
    print("\n Nombre de lignes dupliquées :", df.duplicated().sum())

    # Statistiques descriptives
    print("\n Statistiques descriptives :")
    print(df.describe())
    
    # Vérifier l’équilibre des classes (si une colonne 'target' existe)
    target_column = "target"  # Modifier si besoin
    if target_column in df.columns:
        print(f"\n Distribution des classes dans '{target_column}' :")
        print(df[target_column].value_counts())
    
    # 4. Séparation des données en entraînement et test
    if config.get("train_test_split", False):
        test_size = config.get("test_size", 0.2)
        random_state = config.get("random_state", 42)

        # Vérifier si la colonne 'target' existe pour la séparation
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Séparer les données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            print(f"\n  Données séparées : {100 * (1 - test_size)}% pour l'entraînement, {100 * test_size}% pour le test")
            print(f"   - X_train : {X_train.shape}, X_test : {X_test.shape}")
            print(f"   - y_train : {y_train.shape}, y_test : {y_test.shape}")
        else:
            print("\n  Aucune colonne 'target' trouvée. La séparation train-test n'a pas été effectuée.")
    
    # 5. Sélection et entraînement des modèles
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "SVM": SVC(random_state=random_state),
        "Neural Network": MLPClassifier(random_state=random_state, max_iter=500)
    }

    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Évaluation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Validation croisée
        cross_val = cross_val_score(model, X_train, y_train, cv=5).mean()

        results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Cross-validation score": cross_val
        }

    print("\n Résultats des modèles avant optimisation :")
    for model, metrics in results.items():
        print(f"\n {model}:")
        for metric, value in metrics.items():
            print(f"   - {metric}: {value:.4f}")

    # 6. Optimisation des hyperparamètres avec Optuna
    best_params = {}

    if config["use_optuna"]:
        def objective(trial):
            model_name = trial.suggest_categorical("model", list(models.keys()))

            if model_name == "Decision Tree":
                max_depth = trial.suggest_int("max_depth", 2, 20)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

            elif model_name == "Random Forest":
                n_estimators = trial.suggest_int("n_estimators", 10, 200)
                max_depth = trial.suggest_int("max_depth", 2, 20)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

            elif model_name == "SVM":
                C = trial.suggest_loguniform("C", 0.1, 10)
                kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
                model = SVC(C=C, kernel=kernel, random_state=random_state)

            elif model_name == "Neural Network":
                hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50)])
                learning_rate_init = trial.suggest_loguniform("learning_rate_init", 0.0001, 0.1)
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, max_iter=500, random_state=random_state)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config["n_trials"])

        print("\nMeilleur modèle optimisé avec Optuna :")
        print(study.best_trial)

        best_params = study.best_trial.params

    # 7. Réentraîner les modèles avec les meilleurs hyperparamètres
    if config["retrain_with_best_params"]:
        for model_name in models.keys():
            if model_name in best_params:
                if model_name == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=best_params["max_depth"], random_state=random_state)

                elif model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=random_state)

                elif model_name == "SVM":
                    model = SVC(C=best_params["C"], kernel=best_params["kernel"], random_state=random_state)

                elif model_name == "Neural Network":
                    model = MLPClassifier(hidden_layer_sizes=best_params["hidden_layer_sizes"], learning_rate_init=best_params["learning_rate_init"], max_iter=500, random_state=random_state)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                print(f"\n{model_name} - Après optimisation")
                print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # 8. Sauvegarde du meilleur modèle
    if config["save_best_model"]:
        best_model_name = max(results, key=lambda k: results[k]["Accuracy"])
        best_model = models[best_model_name]

        with open(config["best_model_path"], "wb") as model_file:
            pickle.dump(best_model, model_file)

        print(f"\n Meilleur modèle '{best_model_name}' sauvegardé sous '{config['best_model_path']}'")

   

except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' n'existe pas. Vérifiez le chemin.")
except Exception as e:
    print(f"Erreur lors de l'analyse des données : {e}")
    
