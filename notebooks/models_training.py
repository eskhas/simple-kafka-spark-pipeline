# %% [markdown]
# **Sentiment Analysis of IMDB Movie Reviews**

# %% [markdown]
# **Problem Statement:**
# 
# In this, we have to predict the number of positive and negative reviews based on sentiments by using different classification models.

# %% [markdown]
# **Import necessary libraries**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:53:22.488862Z","iopub.execute_input":"2024-12-20T19:53:22.489176Z","iopub.status.idle":"2024-12-20T19:53:26.005736Z","shell.execute_reply.started":"2024-12-20T19:53:22.489117Z","shell.execute_reply":"2024-12-20T19:53:26.004650Z"}}
#Load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
# **Import the training dataset**

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2024-12-20T19:53:26.007713Z","iopub.execute_input":"2024-12-20T19:53:26.008120Z","iopub.status.idle":"2024-12-20T19:53:27.670389Z","shell.execute_reply.started":"2024-12-20T19:53:26.008045Z","shell.execute_reply":"2024-12-20T19:53:27.669344Z"}}
#importing the training data
imdb_data=pd.read_csv('../input/IMDB Dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)

# %% [markdown]
# **Exploratery data analysis**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:53:27.674487Z","iopub.execute_input":"2024-12-20T19:53:27.674954Z","iopub.status.idle":"2024-12-20T19:53:27.847272Z","shell.execute_reply.started":"2024-12-20T19:53:27.674878Z","shell.execute_reply":"2024-12-20T19:53:27.845121Z"}}
#Summary of the dataset
imdb_data.describe()

# %% [markdown]
# **Sentiment count**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:53:27.850531Z","iopub.execute_input":"2024-12-20T19:53:27.850929Z","iopub.status.idle":"2024-12-20T19:53:27.866620Z","shell.execute_reply.started":"2024-12-20T19:53:27.850877Z","shell.execute_reply":"2024-12-20T19:53:27.865600Z"}}
#sentiment count
imdb_data['sentiment'].value_counts()

# %% [markdown]
# We can see that the dataset is balanced.

# %% [markdown]
# **Spliting the training dataset**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:53:27.868086Z","iopub.execute_input":"2024-12-20T19:53:27.868468Z","iopub.status.idle":"2024-12-20T19:53:27.888760Z","shell.execute_reply.started":"2024-12-20T19:53:27.868392Z","shell.execute_reply":"2024-12-20T19:53:27.887285Z"}}
#split the dataset  
#train dataset
train_reviews=imdb_data.review[:40000]
train_sentiments=imdb_data.sentiment[:40000]
#test dataset
test_reviews=imdb_data.review[40000:]
test_sentiments=imdb_data.sentiment[40000:]
print(train_reviews.shape,train_sentiments.shape)
print(test_reviews.shape,test_sentiments.shape)

# %% [markdown]
# **Text normalization**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:53:27.890300Z","iopub.execute_input":"2024-12-20T19:53:27.890647Z","iopub.status.idle":"2024-12-20T19:53:27.906780Z","shell.execute_reply.started":"2024-12-20T19:53:27.890570Z","shell.execute_reply":"2024-12-20T19:53:27.905848Z"}}
#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

# %% [markdown]
# **Removing html strips and noise text**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:53:27.908473Z","iopub.execute_input":"2024-12-20T19:53:27.908839Z","iopub.status.idle":"2024-12-20T19:53:36.774009Z","shell.execute_reply.started":"2024-12-20T19:53:27.908777Z","shell.execute_reply":"2024-12-20T19:53:36.772796Z"}}
#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(denoise_text)

# %% [markdown]
# **Removing special characters**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:53:36.775413Z","iopub.execute_input":"2024-12-20T19:53:36.775710Z","iopub.status.idle":"2024-12-20T19:53:38.336345Z","shell.execute_reply.started":"2024-12-20T19:53:36.775661Z","shell.execute_reply":"2024-12-20T19:53:38.335257Z"}}
#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_special_characters)

# %% [markdown]
# **Text stemming
# **

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:53:38.338169Z","iopub.execute_input":"2024-12-20T19:53:38.338638Z","iopub.status.idle":"2024-12-20T19:56:57.718804Z","shell.execute_reply.started":"2024-12-20T19:53:38.338477Z","shell.execute_reply":"2024-12-20T19:56:57.717531Z"}}
#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(simple_stemmer)

# %% [markdown]
# **Removing stopwords**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:56:57.720392Z","iopub.execute_input":"2024-12-20T19:56:57.720709Z","iopub.status.idle":"2024-12-20T19:57:43.659167Z","shell.execute_reply.started":"2024-12-20T19:56:57.720658Z","shell.execute_reply":"2024-12-20T19:57:43.658203Z"}}
#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_stopwords)

# %% [markdown]
# **Normalized train reviews**

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-12-20T19:57:43.660474Z","iopub.execute_input":"2024-12-20T19:57:43.660807Z","iopub.status.idle":"2024-12-20T19:57:43.668582Z","shell.execute_reply.started":"2024-12-20T19:57:43.660751Z","shell.execute_reply":"2024-12-20T19:57:43.667615Z"}}
#normalized train reviews
norm_train_reviews=imdb_data.review[:40000]
norm_train_reviews[0]
#convert dataframe to string
#norm_train_string=norm_train_reviews.to_string()
#Spelling correction using Textblob
#norm_train_spelling=TextBlob(norm_train_string)
#norm_train_spelling.correct()
#Tokenization using Textblob
#norm_train_words=norm_train_spelling.words
#norm_train_words

# %% [markdown]
# **Normalized test reviews**

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-12-20T19:57:43.670107Z","iopub.execute_input":"2024-12-20T19:57:43.670408Z","iopub.status.idle":"2024-12-20T19:57:43.681430Z","shell.execute_reply.started":"2024-12-20T19:57:43.670354Z","shell.execute_reply":"2024-12-20T19:57:43.680556Z"}}
#Normalized test reviews
norm_test_reviews=imdb_data.review[40000:]
norm_test_reviews[45005]
##convert dataframe to string
#norm_test_string=norm_test_reviews.to_string()
#spelling correction using Textblob
#norm_test_spelling=TextBlob(norm_test_string)
#print(norm_test_spelling.correct())
#Tokenization using Textblob
#norm_test_words=norm_test_spelling.words
#norm_test_words

# %% [markdown]
# **Bags of words model **
# 
# It is used to convert text documents to numerical vectors or bag of words.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:57:43.683274Z","iopub.execute_input":"2024-12-20T19:57:43.683648Z","iopub.status.idle":"2024-12-20T19:58:53.090644Z","shell.execute_reply.started":"2024-12-20T19:57:43.683574Z","shell.execute_reply":"2024-12-20T19:58:53.089721Z"}}
#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
#transformed train reviews
cv_train_reviews=cv.fit_transform(norm_train_reviews)
#transformed test reviews
cv_test_reviews=cv.transform(norm_test_reviews)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)
#vocab=cv.get_feature_names()-toget feature names

# %% [markdown]
# **Term Frequency-Inverse Document Frequency model (TFIDF)**
# 
# It is used to convert text documents to  matrix of  tfidf features.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T19:58:53.092321Z","iopub.execute_input":"2024-12-20T19:58:53.092708Z","iopub.status.idle":"2024-12-20T20:00:08.331703Z","shell.execute_reply.started":"2024-12-20T19:58:53.092644Z","shell.execute_reply":"2024-12-20T20:00:08.330748Z"}}
#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tv_train_reviews=tv.fit_transform(norm_train_reviews)
#transformed test reviews
tv_test_reviews=tv.transform(norm_test_reviews)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)

# %% [markdown]
# **Labeling the sentiment text**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:08.333576Z","iopub.execute_input":"2024-12-20T20:00:08.333876Z","iopub.status.idle":"2024-12-20T20:00:08.491526Z","shell.execute_reply.started":"2024-12-20T20:00:08.333830Z","shell.execute_reply":"2024-12-20T20:00:08.490719Z"}}
#labeling the sentient data
lb=LabelBinarizer()
#transformed sentiment data
sentiment_data=lb.fit_transform(imdb_data['sentiment'])
print(sentiment_data.shape)

# %% [markdown]
# **Split the sentiment tdata**

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-12-20T20:00:08.493219Z","iopub.execute_input":"2024-12-20T20:00:08.493493Z","iopub.status.idle":"2024-12-20T20:00:08.498962Z","shell.execute_reply.started":"2024-12-20T20:00:08.493437Z","shell.execute_reply":"2024-12-20T20:00:08.497961Z"}}
#Spliting the sentiment data
train_sentiments=sentiment_data[:40000]
test_sentiments=sentiment_data[40000:]
print(train_sentiments)
print(test_sentiments)

# %% [markdown]
# **Modelling the dataset**

# %% [markdown]
# Let us build logistic regression model for both bag of words and tfidf features

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:08.500578Z","iopub.execute_input":"2024-12-20T20:00:08.500931Z","iopub.status.idle":"2024-12-20T20:00:36.157773Z","shell.execute_reply.started":"2024-12-20T20:00:08.500862Z","shell.execute_reply":"2024-12-20T20:00:36.156629Z"}}
#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_reviews,train_sentiments)
print(lr_bow)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_reviews,train_sentiments)
print(lr_tfidf)

# %% [markdown]
# **Logistic regression model performane on test dataset**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:36.159218Z","iopub.execute_input":"2024-12-20T20:00:36.159455Z","iopub.status.idle":"2024-12-20T20:00:36.173069Z","shell.execute_reply.started":"2024-12-20T20:00:36.159415Z","shell.execute_reply":"2024-12-20T20:00:36.171953Z"}}
#Predicting the model for bag of words
lr_bow_predict=lr.predict(cv_test_reviews)
print(lr_bow_predict)
##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(tv_test_reviews)
print(lr_tfidf_predict)

# %% [markdown]
# **Accuracy of the model**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:36.174809Z","iopub.execute_input":"2024-12-20T20:00:36.175150Z","iopub.status.idle":"2024-12-20T20:00:36.184869Z","shell.execute_reply.started":"2024-12-20T20:00:36.175088Z","shell.execute_reply":"2024-12-20T20:00:36.183719Z"}}
#Accuracy score for bag of words
lr_bow_score=accuracy_score(test_sentiments,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)
#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)

# %% [markdown]
# **Print the classification report**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:36.186415Z","iopub.execute_input":"2024-12-20T20:00:36.186792Z","iopub.status.idle":"2024-12-20T20:00:36.240063Z","shell.execute_reply.started":"2024-12-20T20:00:36.186729Z","shell.execute_reply":"2024-12-20T20:00:36.239065Z"}}
#Classification report for bag of words 
lr_bow_report=classification_report(test_sentiments,lr_bow_predict,target_names=['Positive','Negative'])
print(lr_bow_report)

#Classification report for tfidf features
lr_tfidf_report=classification_report(test_sentiments,lr_tfidf_predict,target_names=['Positive','Negative'])
print(lr_tfidf_report)

# %% [markdown]
# **Confusion matrix**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:36.245182Z","iopub.execute_input":"2024-12-20T20:00:36.245435Z","iopub.status.idle":"2024-12-20T20:00:36.272854Z","shell.execute_reply.started":"2024-12-20T20:00:36.245392Z","shell.execute_reply":"2024-12-20T20:00:36.271704Z"}}
#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,lr_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,lr_tfidf_predict,labels=[1,0])
print(cm_tfidf)

# %% [markdown]
# **Stochastic gradient descent or Linear support vector machines for bag of words and tfidf features**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:36.274642Z","iopub.execute_input":"2024-12-20T20:00:36.275002Z","iopub.status.idle":"2024-12-20T20:00:39.304641Z","shell.execute_reply.started":"2024-12-20T20:00:36.274938Z","shell.execute_reply":"2024-12-20T20:00:39.303363Z"}}
#training the linear svm
svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)
#fitting the svm for bag of words
svm_bow=svm.fit(cv_train_reviews,train_sentiments)
print(svm_bow)
#fitting the svm for tfidf features
svm_tfidf=svm.fit(tv_train_reviews,train_sentiments)
print(svm_tfidf)

# %% [markdown]
# **Model performance on test data**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:39.307690Z","iopub.execute_input":"2024-12-20T20:00:39.308079Z","iopub.status.idle":"2024-12-20T20:00:39.333346Z","shell.execute_reply.started":"2024-12-20T20:00:39.308020Z","shell.execute_reply":"2024-12-20T20:00:39.329469Z"}}
#Predicting the model for bag of words
svm_bow_predict=svm.predict(cv_test_reviews)
print(svm_bow_predict)
#Predicting the model for tfidf features
svm_tfidf_predict=svm.predict(tv_test_reviews)
print(svm_tfidf_predict)

# %% [markdown]
# **Accuracy of the model**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:39.335378Z","iopub.execute_input":"2024-12-20T20:00:39.338823Z","iopub.status.idle":"2024-12-20T20:00:39.365515Z","shell.execute_reply.started":"2024-12-20T20:00:39.338734Z","shell.execute_reply":"2024-12-20T20:00:39.364284Z"}}
#Accuracy score for bag of words
svm_bow_score=accuracy_score(test_sentiments,svm_bow_predict)
print("svm_bow_score :",svm_bow_score)
#Accuracy score for tfidf features
svm_tfidf_score=accuracy_score(test_sentiments,svm_tfidf_predict)
print("svm_tfidf_score :",svm_tfidf_score)

# %% [markdown]
# **Print the classification report**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:39.367269Z","iopub.execute_input":"2024-12-20T20:00:39.367715Z","iopub.status.idle":"2024-12-20T20:00:39.442149Z","shell.execute_reply.started":"2024-12-20T20:00:39.367636Z","shell.execute_reply":"2024-12-20T20:00:39.441129Z"}}
#Classification report for bag of words 
svm_bow_report=classification_report(test_sentiments,svm_bow_predict,target_names=['Positive','Negative'])
print(svm_bow_report)
#Classification report for tfidf features
svm_tfidf_report=classification_report(test_sentiments,svm_tfidf_predict,target_names=['Positive','Negative'])
print(svm_tfidf_report)

# %% [markdown]
# **Plot the confusion matrix**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:39.444240Z","iopub.execute_input":"2024-12-20T20:00:39.444671Z","iopub.status.idle":"2024-12-20T20:00:39.473055Z","shell.execute_reply.started":"2024-12-20T20:00:39.444589Z","shell.execute_reply":"2024-12-20T20:00:39.471931Z"}}
#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,svm_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,svm_tfidf_predict,labels=[1,0])
print(cm_tfidf)

# %% [markdown]
# **Multinomial Naive Bayes for bag of words and tfidf features**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:39.475092Z","iopub.execute_input":"2024-12-20T20:00:39.475466Z","iopub.status.idle":"2024-12-20T20:00:41.617205Z","shell.execute_reply.started":"2024-12-20T20:00:39.475398Z","shell.execute_reply":"2024-12-20T20:00:41.616032Z"}}
#training the model
mnb=MultinomialNB()
#fitting the svm for bag of words
mnb_bow=mnb.fit(cv_train_reviews,train_sentiments)
print(mnb_bow)
#fitting the svm for tfidf features
mnb_tfidf=mnb.fit(tv_train_reviews,train_sentiments)
print(mnb_tfidf)

# %% [markdown]
# **Model performance on test data**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:41.618757Z","iopub.execute_input":"2024-12-20T20:00:41.619017Z","iopub.status.idle":"2024-12-20T20:00:41.918190Z","shell.execute_reply.started":"2024-12-20T20:00:41.618974Z","shell.execute_reply":"2024-12-20T20:00:41.917121Z"}}
#Predicting the model for bag of words
mnb_bow_predict=mnb.predict(cv_test_reviews)
print(mnb_bow_predict)
#Predicting the model for tfidf features
mnb_tfidf_predict=mnb.predict(tv_test_reviews)
print(mnb_tfidf_predict)

# %% [markdown]
# **Accuracy of the model**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:41.919514Z","iopub.execute_input":"2024-12-20T20:00:41.919778Z","iopub.status.idle":"2024-12-20T20:00:41.928027Z","shell.execute_reply.started":"2024-12-20T20:00:41.919737Z","shell.execute_reply":"2024-12-20T20:00:41.927040Z"}}
#Accuracy score for bag of words
mnb_bow_score=accuracy_score(test_sentiments,mnb_bow_predict)
print("mnb_bow_score :",mnb_bow_score)
#Accuracy score for tfidf features
mnb_tfidf_score=accuracy_score(test_sentiments,mnb_tfidf_predict)
print("mnb_tfidf_score :",mnb_tfidf_score)

# %% [markdown]
# **Print the classification report**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:41.929244Z","iopub.execute_input":"2024-12-20T20:00:41.929465Z","iopub.status.idle":"2024-12-20T20:00:41.982146Z","shell.execute_reply.started":"2024-12-20T20:00:41.929425Z","shell.execute_reply":"2024-12-20T20:00:41.981231Z"}}
#Classification report for bag of words 
mnb_bow_report=classification_report(test_sentiments,mnb_bow_predict,target_names=['Positive','Negative'])
print(mnb_bow_report)
#Classification report for tfidf features
mnb_tfidf_report=classification_report(test_sentiments,mnb_tfidf_predict,target_names=['Positive','Negative'])
print(mnb_tfidf_report)

# %% [markdown]
# **Plot the confusion matrix**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:41.983953Z","iopub.execute_input":"2024-12-20T20:00:41.984316Z","iopub.status.idle":"2024-12-20T20:00:42.011352Z","shell.execute_reply.started":"2024-12-20T20:00:41.984252Z","shell.execute_reply":"2024-12-20T20:00:42.010527Z"}}
#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,mnb_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,mnb_tfidf_predict,labels=[1,0])
print(cm_tfidf)

# %% [markdown]
# **Let us see positive and negative words by using WordCloud.**

# %% [markdown]
# **Word cloud for positive review words**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:42.012925Z","iopub.execute_input":"2024-12-20T20:00:42.013197Z","iopub.status.idle":"2024-12-20T20:00:42.949355Z","shell.execute_reply.started":"2024-12-20T20:00:42.013144Z","shell.execute_reply":"2024-12-20T20:00:42.948323Z"}}
#word cloud for positive review words
plt.figure(figsize=(10,10))
positive_text=norm_train_reviews[1]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
positive_words=WC.generate(positive_text)
plt.imshow(positive_words,interpolation='bilinear')
plt.show

# %% [markdown]
# **Word cloud for negative review words**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-20T20:00:42.950971Z","iopub.execute_input":"2024-12-20T20:00:42.951288Z","iopub.status.idle":"2024-12-20T20:00:43.952875Z","shell.execute_reply.started":"2024-12-20T20:00:42.951240Z","shell.execute_reply":"2024-12-20T20:00:43.951433Z"}}
#Word cloud for negative review words
plt.figure(figsize=(10,10))
negative_text=norm_train_reviews[8]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
negative_words=WC.generate(negative_text)
plt.imshow(negative_words,interpolation='bilinear')
plt.show

# %% [markdown]
# **Conclusion:**
# * We can observed that both logistic regression and multinomial naive bayes model performing well compared to linear support vector  machines.
# * Still we can improve the accuracy of the models by preprocessing data and by using lexicon models like Textblob.