##############################################################################
### Module simple implémentant les deux métriques décrites dans le rapport ###
##############################################################################

import numpy
import pandas

def _compute_protected_proportions(protected_array,prediction_array):
    """Fonction pour calculer les proportions de prédiction postives dans chaque classe"""

    results = {}
    for k,race in enumerate(protected_array): # Pour chaque prédiction, on compte le nombre de positifs dans chaque classe.

        if prediction_array[k] == 1:

            if race in results.keys():
                count = results[race]
                results[race] = count + 1
            else:
                results[race] = 1
    
    race_count = protected_array.value_counts()

    for race in results.keys():
        proportion = results[race] / race_count[race] # On enregistre la proportion finale.
        results[race] = proportion
    
    return results


def _find_minmax_proportions(proportion_dict):
    """Simple parcours de la liste des proportions pour trouver le min/max et les labels associés"""
    prop_min = 1
    prop_max = 0
    for race, prop in proportion_dict.items():
        if prop <= prop_min:
            prop_min = prop
            race_min = race
        if prop >= prop_max:
            prop_max = prop
            race_max = race
    
    groups_considered = (race_max,race_min)
    
    return prop_max, prop_min, groups_considered

def intersectional_demographic_parity(y_pred, protected_test):
    """Calcul de la DDP dans sa version multiclasse la plus conservatrice (min/max)"""

    proportion_dict = _compute_protected_proportions(protected_test,y_pred)
    prop_max, prop_min, groups_considered = _find_minmax_proportions(proportion_dict)

    return prop_max - prop_min, groups_considered

def intersectional_p_percent(y_pred, protected_test):
    """Calcul de la p%-rule selon le même principe de généralisation min/max"""

    proportion_dict = _compute_protected_proportions(protected_test,y_pred)
    prop_max, prop_min, groups_considered = _find_minmax_proportions(proportion_dict)

    return prop_min / prop_max, groups_considered