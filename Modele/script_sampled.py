import pandas as pd
import json
import optuna
import pickle
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
        df = pd.read_csv(file_path)
    elif file_type == "json":
        df = pd.read_json(file_path)
    elif file_type == "excel":
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Format non supporté : 'csv', 'json' ou 'excel'.")

    print("Données chargées avec succès.")

    # 3. Appliquer l'échantillonnage si activé dans la configuration
    if config.get("sampling", False):
        fraction = config.get("sampling_fraction", 0.2)
        df = df.sample(frac=fraction, random_state=42)
        print(f"\n Échantillonnage activé : {fraction*100}% des données utilisées.")

    # 4. Comprendre les données échantillonnées

    # Aperçu des données
    print("\n Aperçu des données échantillonnées :")
    print(df.head())

    # Informations générales
    df.info()

    # Dimensions du dataset
    print("\n Dimensions du dataset :", df.shape)

    # Types de données
    print("\n Types de données par colonne :")
    print(df.dtypes)

    # Valeurs manquantes
    print("\n Valeurs manquantes par colonne :")
    print(df.isnull().sum())

    # Valeurs uniques par colonne
    print("\n Nombre de valeurs uniques par colonne :")
    print(df.nunique())

    # Détection des doublons
    print("\n Nombre de lignes dupliquées :", df.duplicated().sum())

    # Statistiques descriptives
    print("\n Statistiques descriptives :")
    print(df.describe())

    # Vérification de l’équilibre des classes si la colonne "target" existe
    target_column = "target"
    if target_column in df.columns:
        print(f"\n Distribution des classes dans '{target_column}' :")
        print(df[target_column].value_counts())
    
    # 5. Séparation des données en entraînement et test si activé dans la configuration
    if config.get("train_test_split", False):
        test_size = config.get("test_size", 0.2)
        random_state = config.get("random_state", 42)

        # Vérifier si la colonne 'target' existe pour la séparation
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Séparer les données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            print(f"\n Données séparées : {100 * (1 - test_size)}% pour l'entraînement, {100 * test_size}% pour le test")
            print(f"   - X_train : {X_train.shape}, X_test : {X_test.shape}")
            print(f"   - y_train : {y_train.shape}, y_test : {y_test.shape}")
        else:
            print("\n Aucune colonne 'target' trouvée. La séparation train-test n'a pas été effectuée.")
            
    # 6. Sélection et entraînement des modèles
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

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

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

    # 7. Optimisation des hyperparamètres avec Optuna
    best_params = {}
    best_model = None
    best_score = 0

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

    # 8. Réentraîner les modèles avec les meilleurs hyperparamètres et sauvegarde du modèle
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

                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model

                print(f"\n{model_name} - Après optimisation")
                print(f"Accuracy: {accuracy:.4f}")

    # 9. Sauvegarde du meilleur modèle
    if config["save_best_model"] and best_model is not None:
        with open(config["model_save_path"], "wb") as f:
            pickle.dump(best_model, f)
        print(f"\nMeilleur modèle sauvegardé sous {config['model_save_path']}")

except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' n'existe pas.")
except Exception as e:
    print(f"Erreur lors de l'analyse des données : {e}")
