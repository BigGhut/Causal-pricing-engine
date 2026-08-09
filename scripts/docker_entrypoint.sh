#!/bin/sh
set -e

DPE_DB="${CPE_DPE__DB_PATH:-/app/data/dpe_database.db}"

if [ "${CPE_RETRAIN_ON_START:-0}" = "1" ] && [ -f "$DPE_DB" ]; then
    echo "[Docker Entrypoint] Retraining model on DPE dataset at $DPE_DB..."
    python scripts/train.py --source dpe --dpe-db-path "$DPE_DB" || echo "[Docker Entrypoint] Retrain failed, keeping fallback artifact."
elif [ -f "$DPE_DB" ] && [ ! -f "artifacts/model.joblib" ]; then
    echo "[Docker Entrypoint] Artifact missing. Retraining model on DPE dataset..."
    python scripts/train.py --source dpe --dpe-db-path "$DPE_DB" || echo "[Docker Entrypoint] Retrain failed."
fi

exec "$@"
