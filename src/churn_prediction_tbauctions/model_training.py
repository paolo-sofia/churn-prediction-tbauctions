import pandas as pd
import polars as pl
import random
import pathlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import mlflow
import gc
import plotly.express as px
import shap

import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, ConfusionMatrixDisplay, \
    PrecisionRecallDisplay, RocCurveDisplay
from sklearn.compose import make_column_transformer
import logging


pl.Config(set_fmt_float="full")
pd.options.display.float_format = '{:.3f}'.format
# pd.options.plotting.backend = "matplotlib"

mlflow.set_tracking_uri("../data/mlflow_runs")

INPUT_DATA_PATH = pathlib.Path("../data/processed.parquet")
SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def retrieve_categorical_columns_to_encode(dataframe: pl.LazyFrame) -> tuple[list[str], list[str]]:
    one_hot_encoded_columns: list[str] = []
    ordinal_encoded_columns: list[str] = []

    for col in dataframe.select(pl.col(pl.Categorical)).columns:
        num_categories = len(dataframe.select(pl.col(col).cat.get_categories()).collect().to_series().to_list())
        if num_categories <= 5:
            one_hot_encoded_columns.append(col)
        else:
            ordinal_encoded_columns.append(col)

    return one_hot_encoded_columns, ordinal_encoded_columns


def retrieve_numerical_columns_to_encode(dataframe: pl.LazyFrame) -> list[str]:
    return dataframe.select(pl.col(pl.NUMERIC_DTYPES)).columns


def store_metrics(y_valid: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series, valid_sample_weight: np.ndarray) -> \
dict[str, float]:
    fig = ConfusionMatrixDisplay.from_predictions(y_true=y_valid, y_pred=y_pred, normalize="all",
                                                  sample_weight=valid_sample_weight)
    mlflow.log_figure(fig.figure_, artifact_file="confusion_matrix.png")

    fig = PrecisionRecallDisplay.from_predictions(y_true=y_valid, y_pred=y_pred_proba,
                                                  sample_weight=valid_sample_weight)
    mlflow.log_figure(fig.figure_, artifact_file="precision_recall_curve.png")

    fig = RocCurveDisplay.from_predictions(y_true=y_valid, y_pred=y_pred_proba, sample_weight=valid_sample_weight)
    mlflow.log_figure(fig.figure_, artifact_file="roc_curve.png")

    conf_matrix: np.ndarray = confusion_matrix(y_pred=y_pred, y_true=y_valid, labels=[0, 1],
                                               sample_weight=valid_sample_weight)
    tn, fp, fn, tp = conf_matrix.ravel()

    epsilon = 1e-15

    # Calculate metrics
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    balanced_accuracy = (recall + specificity) / 2
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    false_positive_rate = fp / (fp + tn + epsilon)
    false_negative_rate = fn / (fn + tp + epsilon)
    mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + epsilon)
    fowlkes_mallows_index = tp / (((tp + fp) * (tp + fn)) ** 0.5 + epsilon)
    roc_auc = roc_auc_score(y_true=y_valid, y_score=y_pred_proba, sample_weight=valid_sample_weight)
    average_precision = average_precision_score(y_true=y_valid, y_score=y_pred_proba, sample_weight=valid_sample_weight)

    metrics = {
        "accuracy": round(accuracy, 3),
        "balanced_accuracy": round(balanced_accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "specificity": round(specificity, 3),
        "f1": round(f1_score, 3),
        "false_positive_rate": round(false_positive_rate, 3),
        "false_negative_rate": round(false_negative_rate, 3),
        "average_precision": round(average_precision, 3),
        "mcc": round(mcc, 3),
        "fowlkes_mallows_index": round(fowlkes_mallows_index, 3),
        "auc": round(roc_auc, 3),
    }

    mlflow.log_metrics(metrics)

    return metrics


def compute_feature_importances(model, X_train, X_valid) -> pd.DataFrame:
    explainer = shap.LinearExplainer(model, X_train, nsamples=100_000, seed=SEED)
    shap_values = explainer.shap_values(X_valid)
    feature_importances: pd.DataFrame = pd.DataFrame(
        data={"column": list(X_valid.columns), "importance": np.mean(shap_values, axis=0)}).sort_values(by="importance",
                                                                                                        ascending=True)

    mlflow.log_table(feature_importances, "feature_importances.json")

    fig = px.histogram(feature_importances, x="importance", y="column", orientation="h", width=1500, height=1500)

    # mlflow.log_figure(fig, artifact_file="feature_importances.png")

    return feature_importances


def create_pipeline(one_hot_encoded_columns: list[str], ordinal_encoded_columns: list[str], standardized_columns: list[str]) -> Pipeline:
    column_transformer = make_column_transformer(
        (OneHotEncoder(dtype=np.int32, sparse_output=False, drop="first"), one_hot_encoded_columns),
        (OrdinalEncoder(dtype=np.int32), ordinal_encoded_columns),
        (StandardScaler(), standardized_columns),
        remainder="passthrough",
        n_jobs=-1,
        verbose=1,
        verbose_feature_names_out=False
    ).set_output(transform="polars")

    return column_transformer
    
    # data: pl.DataFrame = column_transformer.fit_transform(data.collect())
    
    
def init_mlflow_experiment(experiment_name: str) -> None:
    mlflow.set_experiment(experiment_name)

    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        log_datasets=False,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        registered_model_name=None,
        extra_tags=None
    )


def train_cross_validation(model, X: pd.DataFrame, y: pd.Series) -> str:
    with mlflow.start_run() as main_run:
        run_id = main_run.info.run_id

        cross_validator: StratifiedKFold = StratifiedKFold(n_splits=5)

        # model: LogisticRegression = LogisticRegression(verbose=0, random_state=SEED, n_jobs=-1)

        metrics = {
            "accuracy": [],
            "balanced_accuracy": [],
            "precision": [],
            "recall": [],
            "specificity": [],
            "f1": [],
            "false_positive_rate": [],
            "false_negative_rate": [],
            "average_precision": [],
            "mcc": [],
            "fowlkes_mallows_index": [],
            "auc": [],
        }
        for i, (train_index, test_index) in enumerate(cross_validator.split(X, y)):
            print(f"Training {i + 1} split")
            X_train, y_train = X.loc[train_index], y.loc[train_index].to_numpy().ravel()
            X_valid, y_valid = X.loc[test_index], y.loc[test_index].to_numpy().ravel()

            train_sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            valid_sample_weights = compute_sample_weight(class_weight="balanced", y=y_valid)
            with mlflow.start_run(nested=True, run_name=f"Split_{i + 1}", log_system_metrics=True):
                mlflow.log_param("columns", list(X_train.columns))
                model.fit(X_train, y_train, sample_weight=train_sample_weights)

                y_pred = model.predict(X_valid)
                y_pred_proba = model.predict_proba(X_valid)[:, 1]

                compute_feature_importances(model=model, X_train=X_train, X_valid=X_valid)

                metrics_split = store_metrics(y_valid=y_valid, y_pred=y_pred, y_pred_proba=y_pred_proba,
                                              valid_sample_weight=valid_sample_weights)
                for metric_key in metrics_split:
                    metrics[metric_key].append(metrics_split[metric_key])

        for metric_name in metrics:
            metric_mean = np.mean(metrics[metric_name])
            metric_std = np.std(metrics[metric_name])
            mlflow.log_metrics({f"{metric_name}_mean": metric_mean, f"{metric_name}_std": metric_std})

    return run_id


def train_full_model(model, X: pd.DataFrame, y: pd.Series, mlflow_run_id: str) -> tuple[LogisticRegression, pd.DataFrame]:
    print("Train full model")
    y: np.ndarray = y.copy(deep=True).to_numpy().ravel()
    model: LogisticRegression = LogisticRegression(verbose=0, random_state=SEED, n_jobs=-1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    train_sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    valid_sample_weights = compute_sample_weight(class_weight="balanced", y=y_valid)
    with mlflow.start_run(run_id=mlflow_run_id, nested=False):
        with mlflow.start_run(nested=True, run_name="Full model", log_system_metrics=True):
            mlflow.log_param("columns", list(X.columns))
            model.fit(X_train, y_train, sample_weight=train_sample_weights)

            y_pred = model.predict(X_valid)
            y_pred_proba = model.predict_proba(X_valid)[:, 1]

            feature_importances = compute_feature_importances(model=model, X_train=X_train, X_valid=X_valid)

            store_metrics(y_valid=y_valid, y_pred=y_pred, y_pred_proba=y_pred_proba,
                          valid_sample_weight=valid_sample_weights)

    return model, feature_importances


def training():
    set_seed(SEED)
    data: pl.LazyFrame = pl.scan_parquet(INPUT_DATA_PATH).drop("last_activity", "ID", "days_since_last_activity")
    standardized_columns: list[str] = retrieve_numerical_columns_to_encode(data)
    one_hot_encoded_columns, ordinal_encoded_columns = retrieve_categorical_columns_to_encode(data)

    pipeline = create_pipeline(
        one_hot_encoded_columns=one_hot_encoded_columns,
        ordinal_encoded_columns=ordinal_encoded_columns,
        standardized_columns=standardized_columns
    )

    data: pl.LazyFrame = data.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    X: pd.DataFrame = data.drop(["churned"]).collect().to_pandas()
    y: pd.Series = data.select("churned").collect().to_pandas()

    X: pl.DataFrame = pipeline.fit_transform(X)

    init_mlflow_experiment(experiment_name="churn_prediction_logistic_regression")

    classifier: LogisticRegression = LogisticRegression(verbose=0, random_state=SEED, n_jobs=-1, warm_start=False)
    mlflow_run_id: str = train_cross_validation(model=classifier, X=X, y=y)

    train_full_model(classifier, X=X, y=y, mlflow_run_id=mlflow_run_id)
