import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import nltk
from nltk.tokenize import ToktokTokenizer
from bs4 import BeautifulSoup
import re

# Make sure to install necessary packages (if not installed yet)
nltk.download('stopwords')
nltk.download('punkt')

# Create the 'data' directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Assuming you've already loaded your dataset
imdb_data = pd.read_csv('data/IMDB_Dataset.csv')

# Normalizing the text (you can add your custom functions for denoising, stemming, stopwords removal, etc.)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    return re.sub(pattern, '', text)

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    return ' '.join([ps.stem(word) for word in text.split()])

def remove_stopwords(text, is_lower_case=False):
    stop = set(stopword_list)
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stop]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stop]
    return ' '.join(filtered_tokens)

# Apply the normalization steps to the reviews
imdb_data['review'] = imdb_data['review'].apply(denoise_text)
imdb_data['review'] = imdb_data['review'].apply(remove_special_characters)
imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)
imdb_data['review'] = imdb_data['review'].apply(remove_stopwords)

# Split the data into training and testing sets
train_reviews = imdb_data.review[:40000]
test_reviews = imdb_data.review[40000:]
train_sentiments = imdb_data.sentiment[:40000]
test_sentiments = imdb_data.sentiment[40000:]

# Vectorize the text data using Bag of Words and TF-IDF
cv = CountVectorizer(min_df=0.0, max_df=1, binary=False, ngram_range=(1, 3))
tv = TfidfVectorizer(min_df=0.0, max_df=1, use_idf=True, ngram_range=(1, 3))

cv_train_reviews = cv.fit_transform(train_reviews)
cv_test_reviews = cv.transform(test_reviews)

tv_train_reviews = tv.fit_transform(train_reviews)
tv_test_reviews = tv.transform(test_reviews)

# Convert sentiments to binary labels
lb = LabelBinarizer()
sentiment_data = lb.fit_transform(imdb_data['sentiment'])

train_sentiments = sentiment_data[:40000]
test_sentiments = sentiment_data[40000:]

# Initialize and train Logistic Regression models
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)

# Train with Bag of Words (BoW) features
lr_bow = lr.fit(cv_train_reviews, train_sentiments)

# Train with TF-IDF features
lr_tfidf = lr.fit(tv_train_reviews, train_sentiments)

# Saving the models and vectorizers
joblib.dump(cv, 'models/cv_vectorizer.joblib')  # Save CountVectorizer model
joblib.dump(tv, 'models/tv_vectorizer.joblib')  # Save TfidfVectorizer model
joblib.dump(lr_bow, 'models/lr_bow_model.joblib')  # Save Logistic Regression model (BoW)
joblib.dump(lr_tfidf, 'models/lr_tfidf_model.joblib')  # Save Logistic Regression model (TFIDF)
joblib.dump(lb, 'models/label_binarizer.joblib')  # Save LabelBinarizer (for sentiment labels)

print("Models and vectorizers saved successfully in the 'models' folder.")
