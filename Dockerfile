FROM apache/airflow:3.0.2

# ffmpeg: cs2_digest's fetch.py shells out to it to downmix audio.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
