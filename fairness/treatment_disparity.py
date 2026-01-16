#################################################################################################################################
### Module implémentant une généralisation de l'algorithme glouton proposé par Lipton et al. au cas multiclasse #################
### L'algorithme procède par "flips" sucessifs des prédictions présentant le meilleur ratio gain de fairness/perte d'accuracy ###
#################################################################################################################################

import numpy as np
from sklearn.linear_model import LogisticRegression


class MulticlassThresholdOptimizer:
    """Entraîne un classifieur satisfaisant une contrainte de parité par un différence de traitement sur les seuils de décision.
        
        En pratique on part d'un classifieur non-contraint et en convertit des prédictions de façon gloutonne à l'aide d'une heuristique
        sur le gain de fairness et la perte d'accuracy."""

    def __init__(self, protected_test, metric='dp'):

        self.model = LogisticRegression(max_iter=1000)
        self.metric = metric

        self.protected = protected_test
        self.categories = np.unique(self.protected)

        self.group_indices = { # On mappe chaque groupe protégé aux indices correspondant à ses membres dans l'array des prédictions
            k: np.where(self.protected == k)[0]
            for k in self.categories
        }

        self.categories_count = { # Taille des groupes
            k: len(self.group_indices[k])
            for k in self.categories
        }

        self.probas = None      # probabilités initiales
        self.y_pred = None     # prédictions globales
        self.proportions = {}  # q_z


    def _fit_base_model(self, X_train, y_train, X_test):
        """Entraîne une régression logistique non-contrainte et calcule les prédictions qui servent de base à l'algorithme de flip."""
        self.model.fit(X_train, y_train)
        self.probas = self.model.predict_proba(X_test)[:, 1]
        self.y_pred = self.model.predict(X_test)


    def _init_proportions(self):
        """Initialise les q_z empiriques à partir des prédictions non-contraintes"""
        self.proportions = {
            k: self.y_pred[self.group_indices[k]].mean()
            for k in self.categories
        }


    def _compute_flip_score(self, i, z, flip_type):
        """
        Calcule le score c_i de chaque prédiction
        i : indice global dans l'array y_pred
        z : nom du groupe
        """
        p_i = self.probas[i]

        if flip_type == 'one_to_zero':
            delta_acc = 2 * p_i - 1
        elif flip_type == 'zero_to_one':
            delta_acc = 1 - 2 * p_i
        else:
            raise ValueError("Unknown flip type")

        return 1 / (self.categories_count[z] * delta_acc)


    def _adjust_thresholds(self, gamma, max_iter=10000):
        """Algorithme glouton qui modifie une par une les prédictions en commençant par celles ayant le meilleur score.
            Ce jusqu'à atteindre l'objectif gamma de parité désiré."""

        self._init_proportions() # On initialise les proportions initiales 
        iter_ = 0

        while True:
            q_vals = np.array(list(self.proportions.values()))
            q_max = q_vals.max()
            q_min = q_vals.min()

            if q_max - q_min <= gamma: # Stopping condition: le seuil de parité démographique est atteint.
                break

            iter_ += 1
            candidates = []

            max_groups = [ # = argmax(q_z) pour z \in Z l'ensemble des groupes considérés.
                z for z in self.categories
                if self.proportions[z] == q_max
            ]
            min_groups = [ # = argmin(q_z)
                z for z in self.categories
                if self.proportions[z] == q_min
            ]

            # Flips 1 -> 0 dans les groupes max
            for z in max_groups:
                for i in self.group_indices[z]:
                    if self.y_pred[i] == 1:
                        score = self._compute_flip_score(i, z, 'one_to_zero') # ATTENTION: on recalcule les scores à chaque itération car les proportions changent.
                        candidates.append((score, z, i))

            # Flips 0 -> 1 dans les groupes min
            for z in min_groups:
                for i in self.group_indices[z]:
                    if self.y_pred[i] == 0:
                        score = self._compute_flip_score(i, z, 'zero_to_one')
                        candidates.append((score, z, i))

            if not candidates: # Stopping condition n°2: pas de candidat valide
                break

            _, z_best, i_best = max(candidates, key=lambda x: x[0])

            # On applique le flip à UN SEUL candidat avec le meilleur score.
            if self.y_pred[i_best] == 1:
                self.y_pred[i_best] = 0
                self.proportions[z_best] -= 1 / self.categories_count[z_best] # ATTENTION: On met à jour les proportions en fonction du flip effectué.
            else:
                self.y_pred[i_best] = 1
                self.proportions[z_best] += 1 / self.categories_count[z_best]

            if iter_ >= max_iter: # Stopping condition n°3: Pas de convergence après un nombre fixé d'itérations.
                print(
                    "Flipping algorithm reached max_iter.\n"
                    f"Achieved demographic parity gap: {q_max - q_min}"
                )
                break

    def fit_transform(self, X_train, y_train, X_test, gamma):
        """Fonction principale poue entraîner le modèle et retourner la prédiction modifiée par les flips.
            -> ATTENTION: C'est la seule fonction qui doit être appelée par l'utilisateur."""
        
        self._fit_base_model(X_train, y_train, X_test)
        self._adjust_thresholds(gamma)
        return self.y_pred