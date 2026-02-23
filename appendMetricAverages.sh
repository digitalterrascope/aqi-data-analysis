#!/usr/bin/env bash
set -euo pipefail

# ===== LOAD ENV FILE =====
ENV_FILE=".env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo ".env file not found"
  exit 1
fi

# export variables from .env
set -o allexport
source "$ENV_FILE"
set +o allexport

# ===== VALIDATE REQUIRED VARS =====
: "${DB_HOST:?Missing DB_HOST}"
: "${DB_PORT:?Missing DB_PORT}"
: "${DB_NAME:?Missing DB_NAME}"
: "${DB_USER:?Missing DB_USER}"
: "${DB_PASSWORD:?Missing DB_PASSWORD}"
DB_TABLE=MetricAverages

CSV_FILE="${1:-metric_avg.csv}"

if [[ ! -f "$CSV_FILE" ]]; then
  echo "CSV file not found: $CSV_FILE"
  exit 1
fi

echo "Loading $CSV_FILE into $DB_TABLE..."

mysql --local-infile=1 \
  -h "$DB_HOST" \
  -P "$DB_PORT" \
  -u "$DB_USER" \
  -p"$DB_PASSWORD" \
  "$DB_NAME" <<EOF

LOAD DATA LOCAL INFILE '$(realpath "$CSV_FILE")'
IGNORE
INTO TABLE $DB_TABLE
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(scrape_id, metric_name, hours, average_value);

EOF

echo "Load complete."