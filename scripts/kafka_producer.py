import json
from kafka import KafkaProducer
import pandas as pd
import time

def produce_messages():
    producer = KafkaProducer(
        bootstrap_servers=kafka_config["bootstrap_servers"],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Load dataset from IMDB_Dataset.csv
    df = pd.read_csv("data/IMDB_Dataset.csv")

    # Convert data to JSON format
    for _, row in df.iterrows():
        record = {"review": row["review"], "label": 1 if row["sentiment"] == "positive" else 0}
        producer.send(kafka_config["topic_name"], value=record)
        print(f"Sent: {record}")
        time.sleep(1)  # Adjust delay if needed

    producer.close()

if __name__ == "__main__":
    from kafka_config import kafka_config
    produce_messages()