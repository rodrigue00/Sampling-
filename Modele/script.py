def combinations(iterable, r):
    """
    Génère des combinaisons de longueur r de l'itérable donné,
    en conservant l'ordre lexicographique.

    :param iterable: La séquence d'entrée (liste, chaîne, etc.)
    :param r: La longueur des combinaisons
    :yield: Un tuple représentant une combinaison unique
    """

    # Convertir l'itérable en tuple pour un accès indexé rapide
    pool = tuple(iterable)
    n = len(pool)
    
    # Si r est plus grand que n, il n'y a pas de combinaison possible
    if r > n:
        return
    
    # Initialiser les premiers indices pour la première combinaison
    # Par exemple, pour r=2 → indices = [0, 1]
    indices = list(range(r))
    
    # Générer la première combinaison avec les indices initiaux
    yield tuple(pool[i] for i in indices)
    
    # Boucle infinie pour générer les combinaisons suivantes
    while True:
        # Trouver l'indice à incrémenter en partant de la fin
        for i in reversed(range(r)):
            # Vérifier si l'indice peut être incrémenté
            if indices[i] != i + n - r:
                break
        else:
            # Si aucun indice ne peut être incrémenté, toutes les combinaisons ont été générées
            return
        
        # Incrémenter l'indice trouvé
        indices[i] += 1
        
        # Réinitialiser les indices suivants pour maintenir l'ordre croissant
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        
        # Générer la nouvelle combinaison avec les indices mis à jour
        yield tuple(pool[k] for k in indices)

# Exemple d'utilisation
if __name__ == "__main__":
    iterable = 'ABCD'
    r = 2
    print(f"Les combinaisons de longueur {r} de '{iterable}' sont :")
    for combo in combinations(iterable, r):
        print(combo)


#########################################################################################
###########################################################

n_wise_features = list(combinations(X.columns, n))

Utilisation de itertools.combinations() :

X.columns : Liste des noms des colonnes (features) du dataset.
combinations(X.columns, n) :
Génère toutes les combinaisons possibles de n colonnes.
Ordre des colonnes non pris en compte → (A, B) est identique à (B, A).
list() : Convertit l'objet combinations en liste pour le réutiliser plusieurs fois.


exemple avec Pairwise (n=2) :

X.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']

n_wise_features = list(combinations(X.columns, 2))
# Résultat :
n_wise_features = [
    ('sepal length', 'sepal width'),
    ('sepal length', 'petal length'),
    ('sepal length', 'petal width'),
    ('sepal width', 'petal length'),
    ('sepal width', 'petal width'),
    ('petal length', 'petal width')
]

6 combinaisons au total pour 4 colonnes :
4 C 2 = 4! / (2! * (4-2)!) = 6


Exemple avec Triplet-wise (n=3) :

n_wise_features = list(combinations(X.columns, 3))
# Résultat :
n_wise_features = [
    ('sepal length', 'sepal width', 'petal length'),
    ('sepal length', 'sepal width', 'petal width'),
    ('sepal length', 'petal length', 'petal width'),
    ('sepal width', 'petal length', 'petal width')
]
4 combinaisons au total pour 4 colonnes :
4 C 3 = 4! / (3! * (4-3)!) = 4

 n_wise_list = []
n_wise_list contiendra toutes les combinaisons N-wise pour chaque ligne du dataset.
Chaque élément de la liste représentera une ligne du dataset avec
toutes les combinaisons possibles pour cette ligne.

for index, row in X.iterrows():

Parcours de chaque ligne du DataFrame X :
index : Indice de la ligne (0, 1, 2, ...).
row : les valeurs des colonnes pour la ligne actuelle.
X.iterrows() :
Itère ligne par ligne sur le DataFrame.
Pour chaque ligne, on obtient :
index : Indice de la ligne.
row : les valeurs des colonnes pour cette ligne.


n_wise stockera toutes les combinaisons N-wise pour la ligne actuelle.
À chaque nouvelle ligne, cette liste est réinitialisée.


for combination in n_wise_features:
    values = tuple(row[feat] for feat in combination)
    n_wise.append(values)
Parcours de toutes les combinaisons possibles (n_wise_features) 
pour cette ligne.
Extraction des valeurs pour cette combinaison :
row[feat] : Valeur de la colonne feat pour la ligne actuelle.
tuple() :
Convertit les valeurs en tuple.
Les tuples sont utilisés comme clé dans un set pour vérifier les combinaisons couvertes.

Exemple avec Pairwise (2-way) :
combination = ('sepal length', 'sepal width')
values = (5.1, 3.5)  # Valeurs pour la première ligne
n_wise.append(values)
# Résultat : n_wise = [(5.1, 3.5)]

n_wise_list.append(n_wise)
Ajoute les combinaisons pour cette ligne à n_wise_list.
n_wise_list :

Exemple pour Pairwise (2-way) :
n_wise_list = [
    [(5.1, 3.5), (5.1, 1.4), (5.1, 0.2), (3.5, 1.4), (3.5, 0.2), (1.4, 0.2)],
    [(4.9, 3.0), (4.9, 1.4), (4.9, 0.2), (3.0, 1.4), (3.0, 0.2), (1.4, 0.2)],
    ...
]

# Fonction pour sélectionner le sous-ensemble minimal couvrant toutes les combinaisons N-wise
def select_minimal_n_wise(n_wise_list):
    covered_combinations = set()
    selected_indices = []

    for idx, combinations in enumerate(n_wise_list):
        new_combinations = set(combinations) - covered_combinations
        if new_combinations:
            selected_indices.append(idx)
            covered_combinations.update(new_combinations)
    
    return selected_indices
    
select_minimal_n_wise() :
Cette fonction permet de sélectionner le sous-ensemble minimal 
d'instances couvrant toutes les combinaisons N-wise.
Objectif :
Réduire le nombre d'instances dans le dataset tout en couvrant
toutes les combinaisons N-wise.
Cela permet de :
Minimiser le nombre de lignes à utiliser pour les tests.
Assurer une couverture maximale des combinaisons.

Pourquoi cette méthode ?
Dans Pairwise (2-way), Triplet-wise (3-way), etc., il y a un grand nombre de combinaisons possibles.
La majorité des lignes du dataset répètent les mêmes combinaisons.
Cette fonction sélectionne un sous-ensemble minimal d'instances pour :
Couvrir toutes les combinaisons.
Éviter les doublons.
Réduire le nombre d'instances.

covered_combinations = set()
selected_indices = []

covered_combinations : 
Un ensemble vide pour stocker toutes les combinaisons déjà couvertes.
On utilise un set car il :
Évite les doublons.
Permet de rechercher et d'ajouter des éléments rapidement.
selected_indices :
Une liste vide pour stocker les indices des lignes sélectionnées.
Ces indices correspondent aux lignes du dataset qui couvrent au moins une nouvelle combinaison.

for idx, combinations in enumerate(n_wise_list):
Parcours de toutes les lignes du dataset :
idx : Indice de la ligne (0, 1, 2, ...).
combinations : Liste des combinaisons N-wise pour cette ligne.
Exemple :

idx = 0
combinations = [(A, B), (A, C), (B, C)]

new_combinations = set(combinations) - covered_combinations
exemple: 
combinations = [(A, B), (A, C), (B, C)]
covered_combinations = {(A, B)}
new_combinations = {(A, C), (B, C)}
La ligne actuelle couvre (A, C) et (B, C) qui ne sont pas encore couvertes.


if new_combinations:
Condition :
Si new_combinations n'est pas vide :
Cela signifie que la ligne actuelle couvre au moins une nouvelle combinaison.
Exemple :
new_combinations = {(A, C), (B, C)} → La ligne couvre de nouvelles combinaisons.
new_combinations = set() → La ligne ne couvre que des combinaisons déjà couvertes.

covered_combinations.update(new_combinations)
Mise à jour des combinaisons couvertes :
covered_combinations.update(new_combinations) :
Ajoute toutes les nouvelles combinaisons à covered_combinations.
Cela évite de sélectionner des lignes redondantes.
Exemple :
new_combinations = {(A, C), (B, C)}
covered_combinations = {(A, B), (A, C), (B, C)}

return selected_indices
Retourne la liste selected_indices :
Contient les indices des lignes sélectionnées pour couvrir toutes les combinaisons.
Cela permet de réduire le dataset tout en conservant toutes les interactions.

for column in X.columns:
        X[column] = pd.qcut(X[column], q=q, labels=[f'Q{i+1}' for i in range(q)])
Pourquoi utiliser pd.qcut() ?
pd.qcut() :
Divise les données en quantiles égaux :
Chaque quantile contient environ le même nombre d'instances.
Les intervalles ne sont pas nécessairement de la même taille.
Adapté aux distributions asymétriques car il :
Équilibre le nombre d'instances dans chaque catégorie.







#####################################################
##################################################"
# ##############################################"
# 
# 

