from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

def train_model():
    spark = SparkSession.builder \
        .appName(spark_config["app_name"] + "_Training") \
        .master(spark_config["master"]) \
        .getOrCreate()

    # Load dataset from IMDB_Dataset.csv
    data = spark.read.csv("data/IMDB_Dataset.csv", header=True, inferSchema=True)
    data = data.withColumn("label", (col("sentiment") == "positive").cast("integer")).drop("sentiment")

    tokenizer = Tokenizer(inputCol="review", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, lr])

    model = pipeline.fit(data)

    print("Model training complete!")
    model.write().overwrite().save("models/sentiment_model")

if __name__ == "__main__":
    from spark_config import spark_config
    train_model()