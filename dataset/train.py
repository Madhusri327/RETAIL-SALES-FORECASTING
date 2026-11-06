import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "archive"


def load_data() -> pd.DataFrame:
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["date"])
    features = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"])
    features = features.drop(columns=["is_holiday"])
    df = sales.merge(
        features,
        on=["store_id", "date"],
        how="left",
        validate="many_to_one",
    )
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df.sort_values("date").reset_index(drop=True)


def make_model():
    categorical_columns = ["store_id", "department", "holiday_name", "season"]
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)],
        remainder="passthrough",
    )
    return make_pipeline(
        preprocessor,
        RandomForestRegressor(
            n_estimators=30,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    )


def make_classifier(feature_columns):
    categorical_columns = ["store_id", "department", "holiday_name", "season"]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_model():
    df = load_data()
    target = df["weekly_sales"]
    X = df.drop(columns=["weekly_sales", "date"])

    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    split_date = dates.iloc[int(len(dates) * 0.8)]
    train_mask = df["date"] < split_date
    test_mask = ~train_mask

    validation_model = make_model()
    validation_model.fit(X.loc[train_mask], target.loc[train_mask])
    predictions = validation_model.predict(X.loc[test_mask])
    actual = target.loc[test_mask]

    # Define a binary classification target using only the training period.
    sales_threshold = float(target.loc[train_mask].median())
    train_classes = (target.loc[train_mask] >= sales_threshold).astype(int)
    test_classes = (target.loc[test_mask] >= sales_threshold).astype(int)
    validation_classifier = make_classifier(X.columns)
    validation_classifier.fit(X.loc[train_mask], train_classes)
    class_predictions = validation_classifier.predict(X.loc[test_mask])

    metrics = {
        "mae": mean_absolute_error(actual, predictions),
        "rmse": np.sqrt(mean_squared_error(actual, predictions)),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "accuracy": accuracy_score(test_classes, class_predictions),
        "precision": precision_score(test_classes, class_predictions, zero_division=0),
        "recall": recall_score(test_classes, class_predictions, zero_division=0),
        "f1_score": f1_score(test_classes, class_predictions, zero_division=0),
        "sales_threshold": sales_threshold,
    }

    model = make_model()
    model.fit(X, target)
    full_sales_threshold = float(target.median())
    classifier = make_classifier(X.columns)
    classifier.fit(X, (target >= full_sales_threshold).astype(int))
    return model, classifier, df, metrics, full_sales_threshold


def main() -> None:
    _, _, _, metrics, _ = train_model()
    print(f"Train rows: {metrics['train_rows']}; test rows: {metrics['test_rows']}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")


if __name__ == "__main__":
    main()