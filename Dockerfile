FROM python:3.11-slim-bookworm
LABEL authors="paolo"

COPY requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./data/DS_interview_case_TBA/ /data/input_data
COPY ./data/models/ /data/models
COPY ./data/columns /data/input_data/
COPY .env /app

COPY ./src/churn_prediction_tbauctions/ /app/

WORKDIR /app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]