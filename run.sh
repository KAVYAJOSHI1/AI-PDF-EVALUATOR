#!/usr/bin/env bash
# Starts the ExamLens web app (UI + API) at http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python   # prefer the project's virtualenv
echo "ExamLens → http://${API_HOST:-127.0.0.1}:${API_PORT:-8000}   (API docs: /docs)"
exec "$PY" -m uvicorn app.api.main:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}" "$@"
