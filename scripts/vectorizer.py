from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CountVectorizerTraining") \
    .getOrCreate()

# Load a static sample of your data
static_data = spark.read.csv("data/IMDB_Dataset.csv", header=True, inferSchema=True)

# Text preprocessing
tokenizer = Tokenizer(inputCol="review", outputCol="words")
words_data = tokenizer.transform(static_data)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filtered_data = remover.transform(words_data)

# Fit CountVectorizer on the static data
vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
vectorizer_model = vectorizer.fit(filtered_data)

# Save the fitted model
vectorizer_model.write().overwrite().save("models/count_vectorizer_model")

spark.stop()