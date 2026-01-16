import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def prepare_data():
    # Noms des colonnes (officiels UCI)
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    # Charger les données
    train = pd.read_csv(
        "data/adult.data",
        names=columns,
        sep=",",
        skipinitialspace=True
    )
    test = pd.read_csv("data/adult.test", names=columns, sep=",", skipinitialspace=True, skiprows=1)

    # Supprimer les valeurs manquantes et les colonnes inutiles
    drop_cols = ["fnlwgt", "education", "native-country"]
    train.drop(columns=drop_cols,inplace=True)
    test.drop(columns=drop_cols,inplace=True)
    train = train.replace("?", pd.NA).dropna()
    test = test.replace("?", pd.NA).dropna()


    # Target binaire
    train["income"] = train["income"].map({"<=50K": 0, ">50K": 1})
    test["income"] = test["income"].map({"<=50K.": 0, ">50K.": 1})

    # Supprimer les colonnes inutiles
    protected_cols = ["income", "race"]
    X_train = train.drop(columns=protected_cols)
    X_test = test.drop(columns=protected_cols)
    
    y_train = train["income"]
    y_test = test["income"]
    protected_train = train['race']
    protected_test = test['race']

    # Colonnes numériques et catégorielles
    num_cols = [
        "age", "education-num",
        "capital-gain", "capital-loss", "hours-per-week"
    ]

    cat_cols = [c for c in X_train.columns if c not in num_cols]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), cat_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor)
    ])

    # Exemple : transformation
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    return X_train, X_test, y_train, y_test, protected_train, protected_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, protected_train, protected_test = prepare_data()