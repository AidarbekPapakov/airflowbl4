FROM apache/airflow:3.0.2
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
