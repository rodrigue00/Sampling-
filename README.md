# 🚀 Efficient Sampling for Machine Learning

_Optimisation de l'échantillonnage pour améliorer l'entraînement des modèles de Machine Learning_

---

## 📌 Introduction

Ce projet s'inscrit dans le cadre de mon **stage en data science** portant sur la thématique **"Efficient Sampling for Machine Learning"**.  
L'objectif principal est d'**optimiser l'échantillonnage des données** afin d'améliorer les performances des modèles de Machine Learning, tout en **réduisant le coût computationnel**.

### 🔹 **Contexte**

Dans de nombreux cas, les datasets utilisés en **Machine Learning** sont volumineux et contiennent des données redondantes ou peu informatives. En appliquant des **techniques d'échantillonnage intelligentes**, on peut :

- **Réduire la taille du dataset** sans compromettre la qualité des modèles.
- **Accélérer l’entraînement** en sélectionnant uniquement les données les plus pertinentes.
- **Optimiser la consommation de ressources** en diminuant le temps de calcul et l’utilisation mémoire.

### 🔹 **Objectif du Projet**

Ce projet propose la mise en place d'un **pipeline de modélisation ML optimisé** qui inclut :

- Une **étape d'échantillonnage** configurable.
- L'entraînement et l'évaluation de plusieurs modèles **(Decision Tree, Random Forest, SVM, MLP)**.
- L'optimisation des hyperparamètres avec **Optuna**.
- La sauvegarde et le déploiement du meilleur modèle.

---

## 📑 Table des Matières

1. [Aperçu du Projet](#-aperçu-du-projet)
2. [Caractéristiques Principales](#-caractéristiques-principales)
3. [Technologies Utilisées](#-technologies-utilisées)
4. [Configuration et Installation](#-configuration-et-installation)
5. [Documentation et Planning](#-documentation-et-planning)
6. [Contributeurs](#-contributeurs)
7. [Licence](#-licence)
8. [Contact](#-contact)

---

## 🌟 Aperçu du Projet

Le projet propose un **pipeline automatisé** de Machine Learning intégrant un mécanisme d'échantillonnage des données et l’optimisation des modèles.  
Voici les **principales fonctionnalités** :

- 📌 **Chargement des données** (avec ou sans échantillonnage).
- ⚙ **Prétraitement et séparation Train/Test**.
- 📊 **Entraînement et comparaison de modèles ML**.
- 🔥 **Optimisation des hyperparamètres avec Optuna**.
- 💾 **Sauvegarde du meilleur modèle** et utilisation pour prédictions.

---

## 🎯 Caractéristiques Principales

✔ **Pipeline flexible** configurable via un fichier `config.json`.  
✔ **Deux scripts d'entraînement** :

- `script_full.py` → **Exécute le pipeline sur l’ensemble des données**.
- `script_sampled.py` → **Exécute le pipeline avec échantillonnage**.  
  ✔ **Optimisation des modèles avec Optuna** pour améliorer les performances.  
  ✔ **Validation croisée** et **comparaison des performances** des modèles.  
  ✔ **Sauvegarde et déploiement du modèle entraîné**.

---

## 🔧 Technologies Utilisées

Le projet repose sur plusieurs bibliothèques Python :

| Technologie            | Description                                  |
| ---------------------- | -------------------------------------------- |
| **Python**             | Langage de programmation principal           |
| **pandas**             | Manipulation et traitement des données       |
| **scikit-learn**       | Entraînement et évaluation des modèles ML    |
| **Optuna**             | Optimisation automatique des hyperparamètres |
| **matplotlib/seaborn** | Visualisation des données                    |
| **pickle**             | Sauvegarde du modèle entraîné                |

---

## ⚙ Configuration et Installation

### 1️⃣ **Cloner le Dépôt**

```bash
git clone https://github.com/utilisateur/nom-du-repo.git
cd nom-du-repo
```
