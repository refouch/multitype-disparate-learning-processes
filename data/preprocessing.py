#################################################################################################
### Fonctions implémentant tout le préprocessing nécéssaire à l'entraînement d'un classifieur ###
#################################################################################################

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path


def prepare_data():
    """Fonction pour importer les données du dataset Adult, supprimer les valeurs inutiles et préparer les données à l'entraînement d'une régression logistique."""

    data_dir = Path(__file__).parent
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    # Charger les données
    train = pd.read_csv(
        train_path,
        names=columns,
        sep=",",
        skipinitialspace=True
    )
    test = pd.read_csv(test_path, names=columns, sep=",", skipinitialspace=True, skiprows=1)

    # Supprimer les valeurs manquantes et les colonnes inutiles
    drop_cols = ["fnlwgt", "education", "native-country"]
    train.drop(columns=drop_cols,inplace=True)
    test.drop(columns=drop_cols,inplace=True)
    train = train.replace("?", pd.NA).dropna()
    test = test.replace("?", pd.NA).dropna()

    # Target binaire
    train["income"] = train["income"].map({"<=50K": 0, ">50K": 1})
    test["income"] = test["income"].map({"<=50K.": 0, ">50K.": 1})

    # Supprimer les colonnes protégées et celle du label
    protected_cols = ["income", "race"]
    X_train = train.drop(columns=protected_cols)
    X_test = test.drop(columns=protected_cols)
    
    y_train = train["income"]
    y_test = test["income"]
    protected_train = train['race']
    protected_test = test['race']

    # Normalisation des colonnes numériques et OneHot-encoding des catégorielles.
    num_cols = [
        "age", "education-num",
        "capital-gain", "capital-loss", "hours-per-week"
    ]

    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), cat_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor)
    ])

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    return X_train, X_test, y_train, y_test, protected_train, protected_test

def prepare_data_with_validation():
    """Même fonction, mais divisant le set d'entraînement pour obtenir un set de 'validation' """

    data_dir = Path(__file__).parent
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    # Charger les données
    train = pd.read_csv(
        train_path,
        names=columns,
        sep=",",
        skipinitialspace=True
    )
    test = pd.read_csv(test_path, names=columns, sep=",", skipinitialspace=True, skiprows=1)

    # Supprimer les valeurs manquantes et les colonnes inutiles
    drop_cols = ["fnlwgt", "education", "native-country"]
    train.drop(columns=drop_cols,inplace=True)
    test.drop(columns=drop_cols,inplace=True)
    train = train.replace("?", pd.NA).dropna()
    test = test.replace("?", pd.NA).dropna()

    # Target binaire
    train["income"] = train["income"].map({"<=50K": 0, ">50K": 1})
    test["income"] = test["income"].map({"<=50K.": 0, ">50K.": 1})

    # Supprimer les colonnes protégées et celle du label
    protected_cols = ["income", "race"]
    X_train = train.drop(columns=protected_cols)
    X_test = test.drop(columns=protected_cols)
    
    y_train = train["income"]
    y_test = test["income"]
    protected_train = train['race']
    protected_test = test['race']

    # Normalisation des colonnes numériques et OneHot-encoding des catégorielles.
    num_cols = [
        "age", "education-num",
        "capital-gain", "capital-loss", "hours-per-week"
    ]

    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), cat_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor)
    ])

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    return X_train, X_test, y_train, y_test, protected_train, protected_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, protected_train, protected_test = prepare_data()