# Multitype Disparate Learning Processes

Ce projet étend les résultats théoriques de **Lipton et al. (NeurIPS 2018)** sur les *Disparate Learning Processes (DLP)* afin d’étudier le cas **d’un attribut protégé multiclasse** (et non plus binaire).

## Objectifs et contributions

L’article de Lipton et al. montre que, sous des contraintes de fairness de type *p%-rule* ou *Calders–Verwer*, la stratégie optimale consiste à appliquer **des seuils de décision spécifiques à chaque groupe protégé**, plutôt qu’à utiliser des DLPs.

Mes contributions personnelles sont les suivantes :
- **Extension formelle et empirique** de ces résultats au cas d’un attribut protégé multiclasse, en utilisant le dataset *UCI Adult* ;
- **Implémentation pratique** d’un algorithme de seuils optimaux par groupe pour la classification binaire sous contrainte de fairness ;
- **Comparaison expérimentale** entre approches de type DLP et seuils explicites multigroupes, en termes de précision, p%-rule, parité démographique et cohérence intra-groupe.

## Structure du dépôt

- `data/` : scripts pour automatiquement télécharger et préparer les données à la tâche de classification
- `fairness/` : impleméntation de l'algorithme des seuils optimaux et des métriques de fairness (p%-rule, demographic parity), tous deux étendus au cas multiclasse.
- `notebooks/` : présentation principale des résultats.

## Reproduire les résultats

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/refouch/multitype-disparate-learning-processes.git
   cd multitype-disparate-learning-processes
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Télécharger le jeu de données:
   ```bash
   python3 data/download_adult.py
   ```

4. Chaque notebook poeut ensuite être utilisé pour reproduire les résultats !

## Référence

> **Lipton, Z. C., Chouldechova, A., & McAuley, J. (2018).**  
> *Does mitigating ML’s impact disparity require treatment disparity?*  
> In *Proceedings of the 32nd Conference on Neural Information Processing Systems (NeurIPS)*.  
> [arXiv:1711.07076](https://arxiv.org/abs/1711.07076)