#!/bin/sh
set -e

# The image trains a synthetic surcharge model at build time. The mounted DPE
# database is a switchback log and is not a training set for that model.
if [ ! -f artifacts/model.joblib ]; then
    echo "[Docker Entrypoint] No artifact. Training the synthetic surcharge model."
    python scripts/train.py --source synthetic
fi

if [ "${CPE_RETRAIN_ON_START:-0}" = "1" ]; then
    echo "[Docker Entrypoint] Retraining the synthetic surcharge model."
    echo "[Docker Entrypoint] DPE switchback logs are not an offer-level experiment."
    python scripts/train.py --source synthetic
fi

exec "$@"
