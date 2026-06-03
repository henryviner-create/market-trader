#!/usr/bin/env bash
#
# Off-box Postgres backup. Run from /opt/market-trader (e.g. daily cron).
#   bash scripts/backup_db.sh [destination_dir]
#
# A backup you haven't restored is not a backup — see scripts/restore_db.sh and
# do a restore drill on a throwaway DB regularly.
set -euo pipefail

DEST="${1:-./backups}"
mkdir -p "$DEST"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
FILE="$DEST/market_trader_${STAMP}.sql.gz"

docker compose exec -T db pg_dump -U "${POSTGRES_USER:-market}" "${POSTGRES_DB:-market_trader}" \
  | gzip > "$FILE"
echo "wrote $FILE ($(du -h "$FILE" | cut -f1))"

# Ship OFF-box (configure ONE — this is the part that makes it a real backup).
# DigitalOcean Spaces (S3-compatible) using the AWS CLI + the .env keys:
#   aws s3 cp "$FILE" "s3://${SPACES_BUCKET}/market-trader/" --endpoint-url "$SPACES_ENDPOINT"
# Or rclone:
#   rclone copy "$FILE" spaces:${SPACES_BUCKET}/market-trader/

# Local retention: keep the most recent 14.
ls -1t "$DEST"/market_trader_*.sql.gz 2>/dev/null | tail -n +15 | xargs -r rm -f
echo "backup complete"
