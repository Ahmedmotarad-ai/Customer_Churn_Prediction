FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /srv

COPY api/requirements.txt ./api/
RUN pip install --no-cache-dir -r ./api/requirements.txt

COPY api/train.py api/app.py ./api/
COPY tasks/Telco_Customer_Churn_Dataset.csv ./tasks/

# Bake a freshly trained model into the image (no stale artifacts)
RUN python ./api/train.py

# Run as non-root at runtime (model file is read-only after build)
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=20s \
    CMD python -c "import urllib.request,os,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000')).getcode()==200 else 1)"

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir /srv/api"]
