FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ training/

# Optional: override at build time with --build-arg DATA_URL=...
ARG DATA_URL
ENV DATA_URL=${DATA_URL}

# Model is built INSIDE the image so no pickle/model binary or dataset
# ever needs to be committed to git. Change DEFAULT_DATA_URL in
# training/train.py (or pass --build-arg DATA_URL=...) to point at the
# real course dataset location.
RUN python training/train.py --out model/model.joblib --metrics-out model/metrics.json

FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY --from=builder /build/model/ model/

ENV MODEL_PATH=/app/model/model.joblib
EXPOSE 8080

# Non-root user for defense-in-depth
RUN useradd -m apiuser
USER apiuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
