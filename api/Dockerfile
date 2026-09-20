FROM python:3.11-slim

WORKDIR /srv

COPY api/requirements.txt ./api/
RUN pip install --no-cache-dir -r ./api/requirements.txt

COPY api/train.py api/app.py ./api/
COPY tasks/Telco_Customer_Churn_Dataset.csv ./tasks/

# Bake a freshly trained model into the image (no stale artifacts)
RUN python ./api/train.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/srv/api"]
