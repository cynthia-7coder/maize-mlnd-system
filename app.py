
from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "mlnd_incidence_model.joblib"
INFO_FILE = BASE_DIR / "model_info.json"
DATA_FILE = BASE_DIR / "Qgis data final.xlsx"

app = Flask(__name__)


def load_model():
    if not MODEL_FILE.exists() or not INFO_FILE.exists():
        return None, None

    return (
        joblib.load(MODEL_FILE),
        json.loads(INFO_FILE.read_text(encoding="utf-8"))
    )


def load_county_analysis():
    """Calculate average recorded incidence and record counts by county."""

    if not DATA_FILE.exists():
        return [], "The Excel dataset could not be found."

    try:
        data = pd.read_excel(DATA_FILE)

        columns = {
            str(column).strip().lower().replace("_", " "): column
            for column in data.columns
        }

        county_column = next(
            (original for name, original in columns.items()
             if name.strip() == "county"),
            None
        )

        incidence_column = next(
            (original for name, original in columns.items()
             if "incidence" in name),
            None
        )

        if county_column is None or incidence_column is None:
            return [], "County or incidence column was not found in the dataset."

        data[incidence_column] = pd.to_numeric(
            data[incidence_column], errors="coerce"
        )

        data = data.dropna(subset=[county_column, incidence_column])
        data[county_column] = data[county_column].astype(str).str.strip()

        analysis = (
            data.groupby(county_column)[incidence_column]
            .agg(["mean", "count"])
            .reset_index()
        )

        analysis.columns = ["county", "average_incidence", "records"]

        county_stats = []

        for _, row in analysis.iterrows():
            county_stats.append({
                "county": str(row["county"]),
                "average_incidence": round(float(row["average_incidence"]), 2),
                "records": int(row["records"])
            })

        county_stats.sort(
            key=lambda item: item["average_incidence"],
            reverse=True
        )

        return county_stats, None

    except Exception as exc:
        return [], f"Could not calculate county analysis: {exc}"


def load_county_risk_distribution():
    """Group recorded field incidence into provisional risk categories."""

    if not DATA_FILE.exists():
        return [], "The Excel dataset could not be found."

    try:
        data = pd.read_excel(DATA_FILE)

        columns = {
            str(column).strip().lower().replace("_", " "): column
            for column in data.columns
        }

        county_column = next(
            (original for name, original in columns.items()
             if name.strip() == "county"),
            None
        )

        incidence_column = next(
            (original for name, original in columns.items()
             if "incidence" in name),
            None
        )

        if county_column is None or incidence_column is None:
            return [], "County or incidence column was not found in the dataset."

        data[incidence_column] = pd.to_numeric(
            data[incidence_column], errors="coerce"
        )

        data = data.dropna(subset=[county_column, incidence_column])
        data[county_column] = data[county_column].astype(str).str.strip()

        def classify_risk(incidence):
            if incidence >= 70:
                return "High"
            elif incidence >= 30:
                return "Moderate"
            return "Low"

        data["risk_category"] = data[incidence_column].apply(classify_risk)

        grouped = (
            data.groupby([county_column, "risk_category"])
            .size()
            .unstack(fill_value=0)
        )

        risk_stats = []

        for county, row in grouped.iterrows():
            risk_stats.append({
                "county": str(county),
                "low": int(row.get("Low", 0)),
                "moderate": int(row.get("Moderate", 0)),
                "high": int(row.get("High", 0)),
                "total": int(row.sum())
            })

        risk_stats.sort(key=lambda item: item["county"])

        return risk_stats, None

    except Exception as exc:
        return [], f"Could not calculate county risk distribution: {exc}"


@app.route("/", methods=["GET", "POST"])
def home():

    model, info = load_model()

    result = None
    error = None

    county_stats, stats_error = load_county_analysis()

    county_risk_stats, risk_stats_error = load_county_risk_distribution()

    if request.method == "POST":

        if model is None:
            error = "The model has not been trained yet. Run train_model.py first."

        else:
            try:
                county = request.form.get("county", "").strip()
                latitude = float(request.form.get("latitude", ""))
                longitude = float(request.form.get("longitude", ""))
                severity = float(request.form.get("severity", ""))

                if county not in info["counties"]:
                    raise ValueError(
                        "Please select a county contained in the dataset."
                    )

                if not -90 <= latitude <= 90:
                    raise ValueError(
                        "Latitude must be between -90 and 90."
                    )

                if not -180 <= longitude <= 180:
                    raise ValueError(
                        "Longitude must be between -180 and 180."
                    )

                if not info["severity_min"] <= severity <= info["severity_max"]:
                    raise ValueError(
                        f"Mean severity must be between "
                        f"{info['severity_min']} and {info['severity_max']}."
                    )

                input_data = pd.DataFrame([{
                    "county": county,
                    "_GPS_latitude": latitude,
                    "_GPS_longitude": longitude,
                    "mean severity": severity,
                }])

                estimate = float(model.predict(input_data)[0])
                estimate = max(0.0, min(100.0, estimate))

                if estimate >= 70:
                    risk = "High"
                    advice = (
                        "Monitor the field closely and contact an "
                        "agricultural extension officer for field assessment."
                    )

                elif estimate >= 30:
                    risk = "Moderate"
                    advice = (
                        "Inspect maize plants regularly and seek agricultural "
                        "guidance if symptoms appear."
                    )

                else:
                    risk = "Low"
                    advice = (
                        "Continue routine field monitoring and follow "
                        "recommended crop hygiene practices."
                    )

                result = {
                    "incidence": round(estimate, 2),
                    "risk": risk,
                    "advice": advice,
                }

            except ValueError as exc:
                error = str(exc)

    return render_template(
        "index.html",
        info=info,
        result=result,
        error=error,
        county_stats=county_stats,
        stats_error=stats_error,
        county_risk_stats=county_risk_stats,
        risk_stats_error=risk_stats_error
    )


if __name__ == "__main__":
    app.run(debug=True)
