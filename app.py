"""Flask dashboard for the local Sentinel Health CKD research demo."""
from __future__ import annotations

import json
import html
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for
from markupsafe import Markup

from sentinel_health.ckd_data import (
    DATA_PATH, FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET, load_dataset,
    make_features, profile_dataset,
)
from sentinel_health.ckd_training import ARTIFACTS, run_training
from sentinel_health.observability import log_prediction, recent_events, telemetry_summary
from sentinel_health.registry import list_versions

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ARTIFACTS / "training_report.json"
PALETTE = {"ckd": "#315f91", "notckd": "#21806d", "amber": "#bd7b25"}
FEATURE_LABELS = {
    "age": "Age (years)", "bp": "Blood pressure (mmHg)", "sg": "Specific gravity",
    "al": "Albumin", "su": "Urine sugar", "rbc": "Red blood cells",
    "pc": "Pus cells", "pcc": "Pus cell clumps", "ba": "Bacteria",
    "bgr": "Random blood glucose (mg/dL)", "bu": "Blood urea (mg/dL)",
    "sc": "Serum creatinine (mg/dL)", "sod": "Sodium (mEq/L)",
    "pot": "Potassium (mEq/L)", "hemo": "Hemoglobin (g/dL)",
    "pcv": "Packed cell volume (%)", "wbcc": "White blood cell count (cells/cmm)",
    "rbcc": "Red blood cell count (millions/cmm)", "htn": "Hypertension",
    "dm": "Diabetes mellitus", "cad": "Coronary artery disease",
    "appet": "Appetite", "pe": "Pedal edema", "ane": "Anemia",
}
NAV = [
    ("/", "Overview"), ("/analytics", "CKD analytics"), ("/quality", "Data quality"),
    ("/prediction", "Prediction demo"), ("/model-lab", "Model lab"),
    ("/explainability", "Explainability"), ("/observability", "Observability"),
    ("/registry", "Model registry"),
]
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


def load_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


def read_report() -> dict | None:
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def load_model():
    path = ARTIFACTS / "ckd_model.joblib"
    return joblib.load(path) if path.exists() else None


def chart(fig, height: int = 340) -> Markup:
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#344b62"),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return Markup(fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False}))


def table(frame: pd.DataFrame, *, index: bool = False, max_rows: int = 100) -> Markup:
    return Markup(frame.head(max_rows).to_html(index=index, classes="data-table", border=0, escape=True))


def metric(label: str, value: str, note: str = "") -> dict:
    return {"label": label, "value": value, "note": note}


def page(title: str, description: str, *, active: str, metrics=None, charts=None,
         tables=None, notice=None, extra=None, error=None, template="page.html"):
    return render_template(template, title=title, description=description, active=active,
                           nav=NAV, metrics=metrics or [], charts=charts or [],
                           tables=tables or [], notice=notice, extra=extra,
                           error=error, model_ready=(ARTIFACTS / "ckd_model.joblib").exists(),
                           row_count=len(load_data()))


@app.get("/")
def overview():
    df = load_data()
    p = profile_dataset(df)
    counts = df[TARGET].value_counts()
    fig = px.bar(pd.DataFrame({"Outcome": ["ckd", "notckd"], "Records": [counts.get("ckd", 0), counts.get("notckd", 0)]}),
                 x="Outcome", y="Records", color="Outcome", text="Records",
                 color_discrete_map=PALETTE)
    return page("Chronic kidney disease classification",
                "A reproducible CKD classification workspace for cohort exploration, model evaluation, and local monitoring.",
                active="/", metrics=[metric("Records", f"{len(df):,}"),
                metric("CKD cases", f"{counts.get('ckd', 0):,}", f"{counts.get('ckd', 0)/len(df):.1%} of cohort"),
                metric("Input features", str(len(FEATURE_COLUMNS))),
                metric("Duplicate records", f"{p['duplicate_rows']:,}")],
                charts=[{"title": "Cohort labels", "html": chart(fig)}],
                extra="<section class='panel'><h2>Workflow</h2><ol><li>Inspect the supplied ARFF and class balance.</li><li>Impute, scale, and encode inside cross-validation folds.</li><li>Compare 30 model and preprocessing configurations.</li><li>Calibrate and select a threshold on validation data.</li><li>Evaluate once on a held-out split and monitor aggregate scoring events.</li></ol><p class='muted'>This historical dataset contains only 400 records. High internal scores do not establish performance at other hospitals or for future patients.</p></section>")


@app.get("/analytics")
def analytics():
    df = load_data()
    class_counts = df[TARGET].value_counts().reindex(["ckd", "notckd"], fill_value=0)
    missing = df[FEATURE_COLUMNS].isna().mean().mul(100).sort_values(ascending=False)
    choices = [c for c in NUMERIC_COLUMNS if df[c].notna().any()]
    selected = request.args.get("feature", choices[0] if choices else "age")
    if selected not in choices:
        selected = choices[0]
    category_choices = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]
    selected_cat = request.args.get("category", category_choices[0])
    if selected_cat not in category_choices:
        selected_cat = category_choices[0]
    missing_by_class = df.groupby(TARGET, observed=False)[FEATURE_COLUMNS].agg(lambda x: x.isna().mean()).mul(100)
    cat = df[[selected_cat, TARGET]].copy()
    cat[selected_cat] = cat[selected_cat].fillna("Missing").astype(str)
    rates = pd.crosstab(cat[selected_cat], cat[TARGET], normalize="index").reindex(columns=["ckd", "notckd"], fill_value=0).reset_index()
    counts = pd.crosstab(cat[selected_cat], cat[TARGET]).reindex(columns=["ckd", "notckd"], fill_value=0).reset_index()
    engineered = make_features(df)
    correlations = df[NUMERIC_COLUMNS].corr(numeric_only=True)
    out_path = ARTIFACTS / "outlier_review.csv"
    if out_path.exists():
        outliers = pd.read_csv(out_path)
    else:
        rows = []
        for col in NUMERIC_COLUMNS:
            q1, q3 = df[col].quantile([.25, .75]); iqr = q3 - q1
            flag = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            rows.append({"feature": col, "iqr_flagged": int(flag.sum()), "rate": float(flag.mean())})
        outliers = pd.DataFrame(rows)
    figs = [
        {"title": "Class distribution", "html": chart(px.bar(class_counts.rename_axis("Outcome").reset_index(name="Records"), x="Outcome", y="Records", color="Outcome", text="Records", color_discrete_map=PALETTE))},
        {"title": "Feature missingness", "html": chart(px.bar(missing.head(16).sort_values().rename("Missing %").rename_axis("Feature").reset_index(), x="Missing %", y="Feature", orientation="h", color="Missing %", color_continuous_scale="Blues"))},
        {"title": "Missingness by outcome", "html": chart(px.imshow(missing_by_class, text_auto=".0f", aspect="auto", color_continuous_scale="Blues", labels={"x": "Input feature", "y": "Outcome", "color": "Missing %"}), 400)},
        {"title": f"{selected}: distribution by class", "html": chart(px.histogram(df, x=selected, color=TARGET, barmode="overlay", opacity=.68, marginal="rug", nbins=24, color_discrete_map=PALETTE))},
        {"title": f"{selected}: class-wise spread", "html": chart(px.box(df, x=TARGET, y=selected, color=TARGET, points="all", color_discrete_map=PALETTE))},
        {"title": f"Class proportions by {selected_cat}", "html": chart(px.bar(rates, x=selected_cat, y=["ckd", "notckd"], barmode="stack", labels={"value": "Within-category share", "variable": "Class"}))},
        {"title": f"Counts by {selected_cat}", "html": chart(px.bar(counts, x=selected_cat, y=["ckd", "notckd"], barmode="group", labels={"value": "Records", "variable": "Class"}))},
        {"title": "Numeric feature correlation", "html": chart(px.imshow(correlations, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), 480)},
        {"title": "IQR outlier screening", "html": chart(px.bar(outliers.sort_values("rate", ascending=False), x="feature", y="rate", color="rate", color_continuous_scale="Oranges", labels={"rate": "Flagged fraction"}))},
    ]
    for col in ["lab_missing_count", "urea_creatinine_ratio"]:
        plot = engineered[[col]].copy(); plot[TARGET] = df[TARGET].to_numpy()
        figs.append({"title": f"Engineered feature: {col}", "html": chart(px.histogram(plot, x=col, color=TARGET, barmode="overlay", opacity=.65, marginal="box", color_discrete_map=PALETTE))})
    summary = df.groupby(TARGET)[NUMERIC_COLUMNS].agg(["count", "median", "mean", "std"]).round(2)
    summary.columns = [f"{a} · {b}" for a, b in summary.columns]
    return page("CKD cohort exploration", "Review target mix, missingness, lab distributions, categorical patterns, outliers, engineered features, and correlations.",
                active="/analytics", metrics=[metric("Records", f"{len(df):,}"), metric("CKD", f"{class_counts['ckd']:,}", f"{class_counts['ckd']/len(df):.1%}"), metric("Not CKD", f"{class_counts['notckd']:,}", f"{class_counts['notckd']/len(df):.1%}"), metric("Fields with missing data", f"{int(df[FEATURE_COLUMNS].isna().any().sum())} / {len(FEATURE_COLUMNS)}")],
                charts=figs, tables=[{"title": "Numeric summary by outcome", "html": table(summary, index=True)}],
                extra=f"<form class='filter-form' method='get'><label>Numeric feature<select name='feature'>{''.join(f'<option value="{c}"{" selected" if c==selected else ""}>{c}</option>' for c in choices)}</select></label><label>Category<select name='category'>{''.join(f'<option value="{c}"{" selected" if c==selected_cat else ""}>{c}</option>' for c in category_choices)}</select></label><button class='button' type='submit'>Update charts</button></form><p class='muted'>IQR flags are for data review only. Valid extreme measurements are retained; they are not clinical abnormality cutoffs.</p>")


@app.get("/quality")
def quality():
    df = load_data()
    rows = [{"Feature": c, "Full name": FEATURE_LABELS.get(c, c), "Type": "numeric" if c in NUMERIC_COLUMNS else "categorical", "Missing": int(df[c].isna().sum()), "Missing rate": f"{df[c].isna().mean():.1%}"} for c in FEATURE_COLUMNS]
    out_path = ARTIFACTS / "outlier_review.csv"
    outliers = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    return page("Data quality", "Audit missing values and review IQR outlier flags. Model imputers are fitted within training folds; no row is automatically removed.", active="/quality",
                tables=[{"title": "Feature completeness", "html": table(pd.DataFrame(rows).sort_values("Missing rate", ascending=False))}, {"title": "IQR screening by numeric measurement", "html": table(outliers.sort_values("rate", ascending=False).assign(rate=lambda x: x.rate.map(lambda v: f"{v:.1%}")) if not outliers.empty else outliers)}],
                notice="IQR flags are not clinical abnormality thresholds. Review flagged values in context; valid extremes remain in the dataset.")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    df, model = load_data(), load_model()
    result, error, submitted_sample = None, None, None
    index = 0
    form_values = {}
    manual_mode = request.method == "POST" and request.form.get("prediction_mode") == "manual"

    if request.method == "POST":
        try:
            if model is None:
                raise ValueError("Train a model in Model lab before requesting a score.")
            if manual_mode:
                form_values = {column: request.form.get(column, "").strip() for column in FEATURE_COLUMNS}
                row = {}
                for column in FEATURE_COLUMNS:
                    value = form_values[column]
                    if not value:
                        row[column] = np.nan
                    elif column in NUMERIC_COLUMNS:
                        try:
                            row[column] = float(value)
                        except ValueError as exc:
                            raise ValueError(f"Enter a valid number for {column}, or leave it blank.") from exc
                    else:
                        allowed = set(df[column].dropna().astype(str).str.lower().unique())
                        normalized = value.lower()
                        if normalized not in allowed:
                            raise ValueError(f"Choose a listed dataset value for {column}, or leave it blank.")
                        row[column] = normalized
                sample = pd.DataFrame([row], columns=FEATURE_COLUMNS)
            else:
                index = int(request.form.get("row_index", "0"))
                if not 0 <= index < len(df):
                    raise ValueError("Choose a row within the bundled dataset.")
                sample = df.iloc[[index]][FEATURE_COLUMNS]
                form_values = {column: "" if pd.isna(sample.iloc[0][column]) else str(sample.iloc[0][column]) for column in FEATURE_COLUMNS}

            started = time.perf_counter()
            probability = float(model.predict_proba(sample)[0, 1])
            elapsed = (time.perf_counter() - started) * 1000
            predicted = "ckd" if probability >= model.threshold else "notckd"
            source_name = "dashboard-manual" if manual_mode else "dashboard-row"
            log_prediction(model_version=model.model_version, probability=probability, threshold=model.threshold,
                           risk_band=predicted, latency_ms=elapsed, source=source_name)
            display_class = "CKD" if predicted == "ckd" else "NOT CKD"
            result = [metric("Predicted class", display_class),
                      metric("Estimated CKD probability", f"{probability:.1%}"),
                      metric("Prediction latency", f"{elapsed:.1f} ms"),
                      metric("Decision threshold", f"{model.threshold:.3f}")]
            submitted_sample = sample
        except (ValueError, TypeError, IndexError) as exc:
            error = str(exc)

    fields = []
    for column in FEATURE_COLUMNS:
        value = html.escape(form_values.get(column, ""), quote=True)
        label = html.escape(FEATURE_LABELS.get(column, column), quote=True)
        if column in NUMERIC_COLUMNS:
            observed = pd.to_numeric(df[column], errors="coerce").dropna()
            low, high = observed.quantile([.05, .95])
            placeholder = html.escape(f"Example range {low:g}-{high:g}", quote=True)
            fields.append(
                f"<label class='prediction-field'><span>{label}</span>"
                f"<input name='{column}' type='number' step='any' value='{value}' placeholder='{placeholder}' aria-label='{label}'>"
                f"</label>"
            )
        else:
            options = sorted(df[column].dropna().astype(str).str.lower().unique())
            option_html = "<option value=''>Unknown / not provided</option>"
            for option in options:
                selected = " selected" if value.lower() == option else ""
                option_html += f"<option value='{html.escape(option, quote=True)}'{selected}>{html.escape(option)}</option>"
            hint = f"Dataset categories: {', '.join(options)}. Leave unknown if not provided."
            fields.append(
                f"<label class='prediction-field'><span>{label}</span>"
                f"<select name='{column}' aria-label='{label}'>{option_html}</select>"
                f"<small>{hint}</small></label>"
            )

    extras = f"""
      <section class='panel manual-prediction'>
        <h2>Enter feature values</h2>
        <p class='muted'>Fill in the fields you know. Example ranges appear inside numeric fields and disappear as you type. Leave unknown values blank. Ranges summarize this dataset; they are not medical reference ranges.</p>
        <form method='post' class='prediction-form'>
          <input type='hidden' name='prediction_mode' value='manual'>
          <div class='prediction-grid'>{''.join(fields)}</div>
          <div class='prediction-actions'><button class='button button-primary' type='submit'>Predict from entered values</button><a class='button' href='{url_for('prediction')}'>Clear fields</a></div>
        </form>
      </section>
      <details class='panel optional-row'>
        <summary>Optional: score a bundled benchmark row</summary>
        <form method='post' class='filter-form'>
          <label>Benchmark row index (0-{len(df)-1})<input name='row_index' type='number' min='0' max='{len(df)-1}' value='{index}' required></label>
          <button class='button' type='submit'>Score example row</button>
        </form>
      </details>
    """
    tables = []
    if submitted_sample is not None:
        missing_count = int(submitted_sample.isna().sum(axis=1).iloc[0])
        displayed_values = submitted_sample.dropna(axis=1, how="all").T.rename(index=FEATURE_LABELS).rename(
            columns={submitted_sample.index[0]: "Value"}
        )
        summary_note = (
            f"<p class='muted'>{missing_count} missing field(s) are hidden here and imputed by the model.</p>"
            if missing_count else "<p class='muted'>All input fields had values.</p>"
        )
        tables.append({"title": "Provided values (missing fields hidden)",
                       "html": Markup(summary_note) + table(displayed_values, index=True)})
    return page("Prediction demo", "Enter feature values for an interactive model prediction. Scoring a bundled benchmark row is optional.",
                active="/prediction", metrics=result or [], tables=tables,
                notice="Educational retrospective dataset demo only. Not a clinical diagnostic tool; do not enter real patient information.",
                error=error, extra=extras)


@app.route("/model-lab", methods=["GET", "POST"])
def model_lab():
    error = None
    trained_message = None
    if request.method == "POST":
        try:
            outcome = run_training(DATA_PATH, ARTIFACTS, cv_folds=5)
            trained_message = f"Training complete. Selected by cross-validation: {outcome['report']['selected_model']}"
        except Exception as exc:
            error = f"Training failed: {type(exc).__name__}: {exc}"
    report = read_report()
    charts, metrics, tables = [], [], []
    extra = "<form method='post'><button class='button button-primary' type='submit'>Train / retrain 30 configurations</button><p class='muted'>Training runs 5-fold stratified validation and can take a few minutes on this local sample.</p></form>"
    if report:
        scores = pd.DataFrame(report["cv_results"]).sort_values("mean_cv_roc_auc")
        charts.append({"title": "30 candidates · mean stratified CV ROC-AUC", "html": chart(px.bar(scores, x="mean_cv_roc_auc", y="candidate", orientation="h", color="mean_cv_roc_auc", color_continuous_scale="Blues"), max(520, len(scores) * 24))})
        m = report["test_metrics"]
        metrics = [metric(k.replace("_", " ").title(), f"{m[k]:.3f}") for k in ["roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"] if k in m]
        tables.append({"title": "Independent holdout metrics", "html": table(pd.DataFrame([{"Metric": k.replace("_", " ").title(), "Value": f"{v:.4f}" if isinstance(v, (int, float)) else str(v)} for k, v in m.items()]))})
        comparison = ARTIFACTS / "model_comparison.csv"
        if comparison.exists():
            tables.append({"title": "All candidate settings", "html": table(pd.read_csv(comparison), max_rows=30)})
        alt = report.get("test_metrics_at_validation_recall_80_threshold", {})
        extra += f"<p class='muted'>Selected {report['selected_model']} using CV only; threshold tuned on validation; held-out test size {report['split']['test']['rows']}. Validation 80%-recall policy on holdout: recall {alt.get('recall', 0):.1%}, precision {alt.get('precision', 0):.1%}, F1 {alt.get('f1', 0):.3f}.</p>"
        artifact_imgs = [("precision_recall.png", "Precision–recall curve"), ("roc_curve.png", "ROC curve"), ("confusion_matrix.png", "Confusion matrix"), ("calibration.png", "Calibration curve")]
        extra += "<div class='image-grid'>" + "".join(f"<figure><img loading='lazy' src='{url_for('artifact', filename=name)}' alt='{label}'><figcaption>{label}</figcaption></figure>" for name, label in artifact_imgs if (ARTIFACTS / name).exists()) + "</div>"
    return page("Thirty model configurations", "Compare preprocessing and hyperparameter variants with stratified cross-validation, select using CV, tune on validation, then evaluate the held-out split.", active="/model-lab", metrics=metrics, charts=charts, tables=tables,
                notice="The source contains only 400 records. Internal scores can be unstable and do not establish clinical performance.", extra=extra, error=error or trained_message)


@app.get("/explainability")
def explainability():
    path = ARTIFACTS / "feature_importance.csv"
    charts = []
    tables = []
    if path.exists():
        imp = pd.read_csv(path).head(20).sort_values("importance")
        charts.append({"title": "Validation permutation importance", "html": chart(px.bar(imp, x="importance", y="feature", orientation="h", error_x="importance_std", color="importance", color_continuous_scale="Blues"), 550)})
        tables.append({"title": "Top feature contributions", "html": table(imp.sort_values("importance", ascending=False))})
    return page("Feature contribution review", "Permutation importance estimates how much shuffling each input changes validation average precision. This is descriptive, not causal.", active="/explainability", charts=charts, tables=tables,
                notice="Confirm every field would be available at the intended prediction time. The retrospective benchmark cannot establish prospective screening performance.")


@app.get("/observability")
def observability():
    tele, report = telemetry_summary(), read_report()
    metrics_report = report["test_metrics"] if report else {}
    offline = report.get("offline_inference_latency", {}) if report else {}
    metrics = [metric("Local requests", f"{tele['requests']:,}", f"{tele['errors']} errors"),
               metric("Live P95 latency", f"{tele['p95_ms']:.1f} ms" if tele["p95_ms"] is not None else "—"),
               metric("Offline P95 latency", f"{offline.get('p95_ms', 0):.1f} ms" if offline else "—"),
               metric("Holdout ROC-AUC", f"{metrics_report['roc_auc']:.3f}" if metrics_report else "—"),
               metric("Holdout Brier score", f"{metrics_report['brier_score']:.3f}" if metrics_report else "—"),
               metric("Error rate", f"{tele['error_rate']:.1%}")]
    tables = []
    drift = ARTIFACTS / "feature_drift.csv"
    if drift.exists(): tables.append({"title": "Training reference vs holdout drift", "html": table(pd.read_csv(drift), max_rows=25)})
    events = recent_events(1000)
    if not events.empty: tables.append({"title": "Recent aggregate scoring events", "html": table(events.drop(columns=["id"], errors="ignore"))})
    return page("Model and service monitoring", "Review holdout discrimination and calibration, feature drift, prediction latency, and aggregate local scoring telemetry.", active="/observability", metrics=metrics, tables=tables,
                notice="Telemetry stores prediction probability, model version, latency, and result metadata only. No row-level input values are stored.")


@app.get("/registry")
def registry():
    state = list_versions()
    versions = state.get("versions", [])
    tables = [{"title": "Registered local model versions", "html": table(pd.DataFrame(versions))}] if versions else []
    card = ARTIFACTS / "model_card.md"
    extra = f"<section class='panel'><h2>Model card</h2><pre>{card.read_text(encoding='utf-8')}</pre></section>" if card.exists() else None
    return page("Model registry", "Inspect locally registered model versions and the active validation threshold.", active="/registry", metrics=[metric("Active version", state.get("active_version", "—"))], tables=tables, extra=extra,
                notice="This local registry is for project demonstration and does not provide clinical model governance.")


@app.get("/artifacts/<path:filename>")
def artifact(filename: str):
    allowed = {"precision_recall.png", "roc_curve.png", "confusion_matrix.png", "calibration.png"}
    if filename not in allowed:
        abort(404)
    return send_from_directory(ARTIFACTS, filename)


@app.get("/health")
def health():
    return {"status": "ok", "application": "sentinel-health", "model_ready": (ARTIFACTS / "ckd_model.joblib").exists()}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(__import__("os").environ.get("PORT", "8501")), debug=False, threaded=True)
