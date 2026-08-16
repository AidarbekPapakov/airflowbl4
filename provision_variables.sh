#!/usr/bin/env bash
# Provisions all Airflow Variables required by csfloat_ingest and
# report_price_changes from this project's .env, against the running
# airflow-apiserver container. Adapted from price_dynamics/airflow-docker/provision.sh.
#
# Neither DAG uses Airflow Connections — ClickHouse (for report_price_changes)
# and SMTP are configured via Variables and docker-compose environment vars
# respectively — so only Variables are provisioned here. csfloat_ingest's
# ClickHouse creds (IFE_CLICKHOUSE_USER/PASSWORD) are NOT set here: they're
# injected as container env vars via docker-compose.yaml instead, since the
# DAG reads them with os.environ, not Variable.get().
#
# Usage: ./provision_variables.sh [path-to-env-file]   (defaults to ./.env)
# Requires: `docker compose up -d` already running (airflow-apiserver healthy).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$SCRIPT_DIR/.env}"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yaml"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: env file not found at $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${IFE_SKINS_FILE_KEY:-}" ]]; then
  echo "Error: IFE_SKINS_FILE_KEY is empty. Retrieve it from item_factor_est's" >&2
  echo "live instance before it's decommissioned:" >&2
  echo "  docker compose -f /home/aidar/projects/item_factor_est/docker-compose.yml \\" >&2
  echo "    exec airflow-apiserver airflow variables get skins_file_key" >&2
  exit 1
fi

# Airflow Variable key -> required env var name in this project's .env
declare -A AIRFLOW_VARS=(
  # item_factor_est (csfloat_ingest)
  [CSFLOAT_API_KEY]=IFE_CSFLOAT_API_KEY
  [CH_HOST_KEY]=IFE_CH_HOST
  [CH_PORT_KEY]=IFE_CH_PORT
  [CH_DB_KEY]=IFE_CH_DB
  [skins_file_key]=IFE_SKINS_FILE_KEY
  # price_dynamics (report_price_changes)
  [steam_inventory_secret]=PD_STEAM_INVENTORY_SECRET
  [csfloat_api_key]=PD_CSFLOAT_API_KEY
  [recipients_email]=PD_RECIPIENTS_EMAIL
  [price_change_threshold_pct]=PD_PRICE_CHANGE_THRESHOLD_PCT
  [comparison_windows_days]=PD_COMPARISON_WINDOWS_DAYS
  [clickhouse_host]=PD_CLICKHOUSE_HOST
  [clickhouse_port]=PD_CLICKHOUSE_PORT
  [clickhouse_user]=PD_CLICKHOUSE_USER
  [clickhouse_password]=PD_CLICKHOUSE_PASSWORD
  [sticker_sp_table]=PD_STICKER_SP_TABLE
  [charm_retention]=PD_CHARM_RETENTION
)

missing=()
for env_name in "${AIRFLOW_VARS[@]}"; do
  if [[ -z "${!env_name:-}" ]]; then
    missing+=("$env_name")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "Error: missing values in $ENV_FILE for: ${missing[*]}" >&2
  exit 1
fi

for airflow_key in "${!AIRFLOW_VARS[@]}"; do
  env_name="${AIRFLOW_VARS[$airflow_key]}"
  echo "Setting Airflow variable: $airflow_key"
  docker compose -f "$COMPOSE_FILE" exec -T airflow-apiserver \
    airflow variables set "$airflow_key" "${!env_name}"
done

echo "Done. Note: IFE_CLICKHOUSE_USER/IFE_CLICKHOUSE_PASSWORD were not set as"
echo "Airflow Variables above — they're injected as container env vars"
echo "(CLICKHOUSE_USER/CLICKHOUSE_PASSWORD) via docker-compose.yaml instead."
