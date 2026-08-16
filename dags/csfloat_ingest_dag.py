from typing import List, Dict, Any
import logging
import os
from datetime import datetime, timedelta

import yaml
from airflow.sdk import dag, task, Variable

from src.ingest.csfloat_client import fetch_sales_history
from src.ingest.clickhouse_loader import insert_sales, validate_and_flatten

log = logging.getLogger(__name__)

SKINS_FOLDER_PATH = os.environ.get(
    "SKINS_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "config"),
)

config: Dict[str, str] = {
    'CSFLOAT_API_KEY': 'CSFLOAT_API_KEY',
    'CH_HOST': 'CH_HOST_KEY',
    'CH_PORT': 'CH_PORT_KEY',
    'CH_DB': 'CH_DB_KEY',
    'skins_file': 'skins_file_key'
}


@dag(
    dag_id="csfloat_ingest",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_tasks=2,  # serialize: concurrent tasks would race on shared rate-limit bucket
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=30),
        "owner": "Aidar",
        "execution_timeout": timedelta(seconds=300) # Timeout task after being idle for >5 minutes
    },
    tags=["csfloat", "skins"],
)
def csfloat_ingest_dag():

    @task()
    def get_skins(**context) -> List[Dict[str, Any]]:

        data_interval_end: datetime = context['data_interval_end']
        day_num: int = int(curr_day) if (isinstance(curr_day := data_interval_end.day, str)) else curr_day

        bathc_file_pattern: str = 'batch'

        batch_files_num: int = len([f for f in os.listdir(SKINS_FOLDER_PATH) if bathc_file_pattern in f])
        batch_file_to_choose: int = day_num%batch_files_num

        skins_file: str = os.path.join(SKINS_FOLDER_PATH, Variable.get(config['skins_file']))
        skins_file = skins_file.format(N=batch_file_to_choose)

        log.info(f"Processing {os.path.split(skins_file)[-1]}")

        with open(skins_file) as f:
            return yaml.safe_load(f)["skins"]

    @task()
    def ingest_skin(skin: Dict[str, Any], **context) -> Dict[str, Any]:
            
        data_interval_end: datetime = context['data_interval_end']
        snapshot_date: str = data_interval_end.strftime("%Y-%m-%d %H:%M:%S")

        API_KEY: str = Variable.get(config['CSFLOAT_API_KEY'])
        CH_HOST: str = Variable.get(config['CH_HOST'])
        CH_PORT: str = Variable.get(config['CH_PORT'])
        CH_DB: str = Variable.get(config['CH_DB'])

        # ClickHouse creds
        CH_USER: str = os.environ.get('CLICKHOUSE_USER')
        CH_PASSWORD: str = os.environ.get('CLICKHOUSE_PASSWORD')

        name = skin["market_hash_name"]
        paint_index = skin.get("paint_index")

        log.info("Fetching sales: %s (paint_index=%s)", name, paint_index)
        raw = fetch_sales_history(name, paint_index, API_KEY)

        if not raw:
            log.info("No sales returned for %s — skipping", name)
            return {"skin": name, "rows_inserted": 0}

        flat = validate_and_flatten(raw, snapshot_date)
        insert_sales(
            flat, 
            host=CH_HOST, port=CH_PORT, database=CH_DB,
            user=CH_USER, password=CH_PASSWORD,    
        )
        log.info("Inserted %d rows for %s", len(flat), name)
        return {"skin": name, "rows_inserted": len(flat)}

    ingest_skin.expand(skin=get_skins())


csfloat_ingest_dag()
