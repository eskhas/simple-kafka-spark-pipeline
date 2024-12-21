from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.classification import LogisticRegressionModel
import joblib
import nltk
import re
from bs4 import BeautifulSoup
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords

# Initialize NLTK resources
nltk.download('stopwords')

# Define preprocessing functions

# Tokenization of text
tokenizer = ToktokTokenizer()
stopword_list = stopwords.words('english')

# Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

# Removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, '', text)

# Text stemming
def simple_stemmer(text):
    ps = nltk.PorterStemmer()
    return ' '.join([ps.stem(word) for word in text.split()])

# Removing stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    return ' '.join(filtered_tokens)

# Preprocessing pipeline function
def preprocess_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    text = simple_stemmer(text)
    return text

# Spark streaming function
def start_stream():
    spark = SparkSession.builder \
        .appName(spark_config["app_name"]) \
        .master(spark_config["master"]) \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.3") \
        .getOrCreate()
    
    # Set log level to ERROR to suppress warnings
    spark.sparkContext.setLogLevel("ERROR")

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

    # Apply preprocessing and prediction
    processed_reviews_df = reviews_df.withColumn("processed_review", udf(preprocess_text, StringType())(col("review")))
    predictions_df = processed_reviews_df.withColumn("prediction", predict_sentiment(col("processed_review")))
    
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
