#!/usr/bin/env bash
#
# Restore a backup. DESTRUCTIVE — overwrites the current database.
#   bash scripts/restore_db.sh backups/market_trader_YYYYMMDDTHHMMSSZ.sql.gz
set -euo pipefail

FILE="${1:?usage: restore_db.sh <backup.sql.gz>}"
echo "Restoring '$FILE' into ${POSTGRES_DB:-market_trader} — this OVERWRITES current data."

gunzip -c "$FILE" | docker compose exec -T db psql -U "${POSTGRES_USER:-market}" "${POSTGRES_DB:-market_trader}"
echo "restore complete — now: docker compose restart engine"
