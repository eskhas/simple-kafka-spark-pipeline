from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover,CountVectorizerModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml import PipelineModel

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

    # # Text preprocessing
    tokenizer = Tokenizer(inputCol="review", outputCol="words")
    tokenized = tokenizer.transform(reviews_df)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    filtered = remover.transform(tokenized)

    # Load pre-trained CountVectorizer model
    vectorizer_model = CountVectorizerModel.load("models/count_vectorizer_model")
    features = vectorizer_model.transform(filtered)

    # Load pre-trained Logistic Regression model
    pipeline_model = PipelineModel.load("models/sentiment_model")

    # Extract the LogisticRegressionModel from the pipeline
    model = pipeline_model.stages[-1] 
    predictions = model.transform(features)

    # Write predictions to console
    query = predictions.select("review", "label", "prediction").writeStream \
        .outputMode("append") \
        .format("console") \
        .option("checkpointLocation", "checkpoints/reviews") \
        .start()

    query.awaitTermination()
if __name__ == "__main__":
    from kafka_config import kafka_config
    from spark_config import spark_config
    preprocess_and_stream()