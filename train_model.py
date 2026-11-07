
from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Qgis data final.xlsx"
MODEL_FILE = BASE_DIR / "mlnd_incidence_model.joblib"
INFO_FILE = BASE_DIR / "model_info.json"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)

    required = [
        "county",
        "_GPS_latitude",
        "_GPS_longitude",
        "mean severity",
        "incidence"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[required].copy()
    df["county"] = df["county"].astype(str).str.strip()

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    if len(df) < 10:
        raise ValueError(
            "At least 10 complete records are required for this prototype."
        )

    X = df[
        ["county", "_GPS_latitude", "_GPS_longitude", "mean severity"]
    ]

    y = df["incidence"]

    categorical_features = ["county"]

    numeric_features = [
        "_GPS_latitude",
        "_GPS_longitude",
        "mean severity"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "county",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric_features
            )
        ]
    )

    # Five-fold cross-validation
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),

        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=4,
            random_state=42
        ),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            random_state=42
        )
    }

    results = []

    best_name = None
    best_pipeline = None
    best_mae = float("inf")

    print("\nEVALUATING MODELS USING 5-FOLD CROSS-VALIDATION")

    for name, estimator in candidates.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        # Generate predictions for every record using folds
        predictions = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv
        )

        mae = mean_absolute_error(y, predictions)

        rmse = mean_squared_error(
            y,
            predictions
        ) ** 0.5

        r2 = r2_score(y, predictions)

        results.append({
            "model": name,
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "r2": round(float(r2), 3)
        })

        print({
            "model": name,
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "r2": round(float(r2), 3)
        })

        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_pipeline = pipeline

    # Train the selected model using all available records
    best_pipeline.fit(X, y)

    joblib.dump(best_pipeline, MODEL_FILE)

    info = {
        "selected_model": best_name,
        "records_used": int(len(df)),
        "counties": sorted(df["county"].unique().tolist()),

        "severity_min": round(
            float(df["mean severity"].min()), 2
        ),

        "severity_max": round(
            float(df["mean severity"].max()), 2
        ),

        "incidence_min": round(
            float(df["incidence"].min()), 2
        ),

        "incidence_max": round(
            float(df["incidence"].max()), 2
        ),

        "evaluation_method": "5-Fold Cross-Validation",
        "cv_folds": 5,
        "evaluation_records": int(len(df)),
        "evaluation": results,
        "target": "incidence",

        "note": (
            "Exploratory estimate of measured MLND incidence, "
            "not a forecast of future outbreaks. "
            "The dataset contains only 45 field records "
            "from four counties. Cross-validation results "
            "are preliminary and may not represent performance "
            "in other locations."
        )
    }

    INFO_FILE.write_text(
        json.dumps(info, indent=4),
        encoding="utf-8"
    )

    print("\nMODEL TRAINING COMPLETE")

    print("Records used:", len(df))
    print("Selected model:", best_name)
    print("Evaluation method: 5-Fold Cross-Validation")

    print("\nFinal evaluation results:")

    for row in results:
        print(row)

    print("\nSaved:", MODEL_FILE.name)
    print("Saved:", INFO_FILE.name)

    print("\nIMPORTANT:", info["note"])


if __name__ == "__main__":
    main()
