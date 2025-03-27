

#  Impact des Techniques de Sampling sur les Performances des Modèles de Machine Learning

> Ce projet explore l’influence des techniques d’échantillonnage sur les performances des modèles de Machine Learning, dans le cadre d’un stage de fin d’études sur le thème **“Efficient Sampling for Machine Learning”**.

---

##  Introduction

Dans le domaine du **Machine Learning**, les datasets peuvent être volumineux et contenir des données redondantes ou peu pertinentes. Ce projet vise à :

- Évaluer l'impact de différentes techniques de sampling sur les performances des modèles.
- Étudier également l'effet des techniques de simplification issues de l'ingénierie logicielle appliquées à l'apprentissage automatique.
- Mettre en place un pipeline ML complet et modulable intégrant des étapes d’échantillonnage, d'entraînement, d’optimisation et de déploiement.

---

##  Contexte

L’utilisation des techniques de sampling permet de :

-  Réduire la taille des datasets.
-  Accélérer l'entraînement des modèles en ne conservant que les données pertinentes.
-  Optimiser la consommation de ressources (temps de calcul, mémoire).

---

##  Table des Matières

1. [Aperçu du Projet](#-aperçu-du-projet)
2. [Technologies Utilisées](#-technologies-utilisées)
4. [Configuration et Installation](#-configuration-et-installation)
5. [Documentation et Planning](#-documentation-et-planning)
6. [Contributeurs](#-contributeurs)
7. [Licence](#-licence)
8. [Auteurs](#-contact)

---

##  Aperçu du Projet

Ce projet propose un pipeline complet de Machine Learning, structuré en plusieurs étapes:

-  Chargement des données (avec ou sans échantillonnage).
-  Prétraitement et split des données (train/test).
-  Entraînement et évaluation de plusieurs modèles.
-  Optimisation automatique avec Optuna.
-  Sauvegarde du modèle final pour une utilisation ultérieure.

---

##  Technologies Utilisées

| Technologie            | Description                                  |
|------------------------|----------------------------------------------|
| **Python**             | Langage de développement principal           |
| **pandas**             | Manipulation des données                     |
| **scikit-learn**       | Entraînement & évaluation de modèles ML      |
| **Optuna**             | Optimisation des hyperparamètres             |
| **matplotlib / seaborn** | Visualisation des résultats                 |
| **pickle**             | Sauvegarde et chargement de modèles          |

---

## Configuration et Installation

 Clonez le référentiel sur votre machine locale :

````frapper
clone git https://github.com/rodrigue00/Sampling-.git
````

 Installez les dépendances requises :

```frapper
pip install -r requirements.txt
```

 Lancer l'Exécution :

```
pip install optuna
pip install xgboost
python script.py

```

---
## Documentation et planning

Documentation : https://www.overleaf.com/project/67ac5bf7920ebce02e127931


---
## Contributeurs
 
Superviseur : Prof. Gilles Perrouin et Dr. Paul Temple

---
##  Licence
 Ce projet est publié publiquement sous les termes de la  [Licence MIT](LICENCE).


---

## Auteurs:
Auteur : Yando rodrigue
Email : rodrigue.yandodjamen@student.unamur.be

---


