import logging

import uvicorn
from fastapi import FastAPI, HTTPException, status
from inference import predict_model
from pydantic import BaseModel

logger = next(logging.getLogger(name) for name in logging.root.manager.loggerDict)
app = FastAPI()


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


@app.get(
    path="/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """Perform a Health Check.

    Endpoint to perform a healthcheck on.

    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


@app.post(path="/predict/{user_id}", tags=["prediction"], summary="Perform a prediction on the churn prediction model")
async def predict_churn(user_id: int) -> dict[str, str]:
    """Perform a prediction on the churn prediction model.

    Args:
        user_id (int): The ID of the user.

    Returns:
        dict[str, str]: A dictionary containing the prediction message for the user, including the prediction result (True or False) and the churn probability.

    Raises:
        HTTPException: If an error occurs while predicting.
    """
    try:
        prediction, pred_proba = predict_model(user_id=user_id)

        if not prediction or not pred_proba:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error while predicting",
            )

        return {"message": f"prediction for user {user_id} is {prediction} with probability of: {pred_proba:.3f}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error while predicting",
        ) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
