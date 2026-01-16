import numpy as np
from sklearn.linear_model import LogisticRegression


class MulticlassThresholdOptimizer:

    def __init__(self, protected_test, metric='p_rule'):
        self.model = LogisticRegression(max_iter=1000)
        self.metric = metric

        self.protected = protected_test
        self.categories = np.unique(self.protected)

        # Indices globaux par groupe
        self.group_indices = {
            k: np.where(self.protected == k)[0]
            for k in self.categories
        }

        # Taille des groupes
        self.categories_count = {
            k: len(self.group_indices[k])
            for k in self.categories
        }

        self.probas = None      # scores globaux
        self.y_pred = None     # prédictions globales
        self.proportions = {}  # q_k


    def _fit_base_model(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        self.probas = self.model.predict_proba(X_test)[:, 1]
        self.y_pred = self.model.predict(X_test)


    def _init_proportions(self):
        """Initialise q_k empiriques"""
        self.proportions = {
            k: self.y_pred[self.group_indices[k]].mean()
            for k in self.categories
        }


    def _compute_flip_score(self, i, z, flip_type):
        """
        i : indice global
        z : groupe
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

        self._init_proportions()
        iter_ = 0

        while True:
            q_vals = np.array(list(self.proportions.values()))
            q_max = q_vals.max()
            q_min = q_vals.min()

            if q_max - q_min <= gamma:
                break

            iter_ += 1
            candidates = []

            max_groups = [
                k for k in self.categories
                if self.proportions[k] == q_max
            ]
            min_groups = [
                k for k in self.categories
                if self.proportions[k] == q_min
            ]

            # Flips 1 -> 0 dans les groupes max
            for z in max_groups:
                for i in self.group_indices[z]:
                    if self.y_pred[i] == 1:
                        score = self._compute_flip_score(i, z, 'one_to_zero')
                        candidates.append((score, z, i))

            # Flips 0 -> 1 dans les groupes min
            for z in min_groups:
                for i in self.group_indices[z]:
                    if self.y_pred[i] == 0:
                        score = self._compute_flip_score(i, z, 'zero_to_one')
                        candidates.append((score, z, i))

            if not candidates:
                break

            _, z_best, i_best = max(candidates, key=lambda x: x[0])

            # Appliquer le flip
            if self.y_pred[i_best] == 1:
                self.y_pred[i_best] = 0
                self.proportions[z_best] -= 1 / self.categories_count[z_best]
            else:
                self.y_pred[i_best] = 1
                self.proportions[z_best] += 1 / self.categories_count[z_best]

            if iter_ >= max_iter:
                print(
                    "Flipping algorithm reached max_iter.\n"
                    f"Achieved demographic parity gap: {q_max - q_min}"
                )
                break

    def fit_transform(self, X_train, y_train, X_test, gamma):
        self._fit_base_model(X_train, y_train, X_test)
        self._adjust_thresholds(gamma)
        return self.y_pred