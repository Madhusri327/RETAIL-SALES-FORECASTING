import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

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
    metrics = {
        "mae": mean_absolute_error(actual, predictions),
        "rmse": np.sqrt(mean_squared_error(actual, predictions)),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
    }

    model = make_model()
    model.fit(X, target)
    return model, df, metrics


def main() -> None:
    _, _, metrics = train_model()
    print(f"Train rows: {metrics['train_rows']}; test rows: {metrics['test_rows']}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")


if __name__ == "__main__":
    main()