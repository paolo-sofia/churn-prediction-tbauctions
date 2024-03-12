import logging
import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline

from src.churn_prediction_tbauctions.preprocess import load_and_preprocess_data_for_inference

load_dotenv()


def load_model() -> Pipeline | None:
    """Load the model for inference.

    Returns:
        Pipeline | None: The loaded model for inference, or None if an error occurs.
    """
    try:
        return pickle.load(open(os.getenv("MODEL_PATH"), "rb"))
    except Exception as e:
        logging.error(e)
        return None


def load_data_pipeline() -> Pipeline | None:
    """Load the data pipeline for inference.

    Returns:
        Pipeline | None: The loaded data pipeline for inference, or None if an error occurs.
    """
    try:
        return pickle.load(open(os.getenv("DATA_PIPELINE_PATH"), "rb"))
    except Exception as e:
        logging.error(e)
        return None


def predict_model(user_id: int) -> tuple[bool | None, float | None]:
    """Predict the churn probability for a given user.

    Args:
        user_id (int): The ID of the user.

    Returns:
        tuple[bool | None, float | None]: A tuple containing the churn prediction (True or False) and the churn probability, or (None, None) if the model or data pipeline is not available.
    """
    model: Pipeline = load_model()
    threshold: float = float(os.getenv("THRESHOLD", "0.5"))
    # data_transform_pipeline = load_data_pipeline()

    if not model:
        return None, None

    data: pd.DataFrame = load_and_preprocess_data_for_inference(user_id=user_id).collect().to_pandas()

    pred_proba = model.predict_proba(data)[:, 1]
    print(f"Prediction proba: {pred_proba}")
    if prediction := pred_proba > threshold:
        return prediction, pred_proba

    return prediction, 1 - pred_proba


if __name__ == "__main__":
    preds = predict_model(user_id=1)
    print(preds)
