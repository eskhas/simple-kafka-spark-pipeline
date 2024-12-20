from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

def preprocess_and_stream():
    spark = SparkSession.builder \
        .appName(spark_config["app_name"]) \
        .master(spark_config["master"]) \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.3") \
        .getOrCreate()

    schema = StructType([
        StructField("review", StringType(), True),
        StructField("label", IntegerType(), True)
    ])

    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_config["bootstrap_servers"]) \
        .option("subscribe", kafka_config["topic_name"]) \
        .load()

    reviews_df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # Preprocessing pipeline
    tokenizer = Tokenizer(inputCol="review", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, lr])

    # Transform the streaming data
    model = pipeline.fit(reviews_df)
    predictions = model.transform(reviews_df)

    query = predictions.select("review", "label", "prediction").writeStream \
        .outputMode("append") \
        .format("console") \
        .option("checkpointLocation", spark_config["checkpoint_dir"] + "/reviews") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    from kafka_config import kafka_config
    from spark_config import spark_config
    preprocess_and_stream()
