from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.classification import LogisticRegressionModel
import joblib

def start_stream():
    spark = SparkSession.builder \
        .appName(spark_config["app_name"]) \
        .master(spark_config["master"]) \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.3") \
        .getOrCreate()

    schema = StructType([
        StructField("review", StringType(), True),
        StructField("label", IntegerType(), True)
    ])

    # Load pre-trained models
    tv_vectorizer = joblib.load("models/tv_vectorizer.joblib")
    lr_model = joblib.load("models/lr_tfidf_model.joblib")

    @udf("double")
    def predict_sentiment(review):
        vectorized = tv_vectorizer.transform([review])
        prediction = lr_model.predict(vectorized)
        return float(prediction[0])

    # Stream data from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_config["bootstrap_servers"]) \
        .option("subscribe", kafka_config["topic_name"]) \
        .load()

    reviews_df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # Apply prediction
    predictions_df = reviews_df.withColumn("prediction", predict_sentiment(col("review")))

    query = predictions_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("checkpointLocation", spark_config["checkpoint_dir"] + "/predictions") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    from kafka_config import kafka_config
    from spark_config import spark_config
    start_stream()
