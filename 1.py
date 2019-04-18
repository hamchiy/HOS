#!/usr/bin/env python
# coding: utf-8

# <img src="../utils/title.png" alt="title", width="600">

# <img src="../utils/master_203_pic.jpg" alt="Master 203", width=400>

# Predicting stock price movements is of clear interest to investors, public companies and governments. There has been a debate on whether the market can be predicted. The Random Walk Theory (Malkiel, 1973) hypothesizes that prices are
# determined randomly and hence it is impossible to outperform the market.
# 
# However, stock price is determined by the behavior of human investors, and the investors determine stock prices by
# using publicly available information to predict how the market will act or react. Financial news
# articles can thus play a large role in influencing the movement of a stock as humans react to the
# information. Previous research has suggested that there is a relationship between news articles
# and stock price movement, as there is some lag between when the news article is released and
# when the market has moved to reflect this information.
# 
# In this lab, we will try to implement a supervised machine learning algorithm to predict Dow Jones index open-close variation given news of the day. This work could be the first step to more sophisticated news analysis systems in order to take decisions on the market.

# ## About data

# The data is composed of more than 5 millions reuters headlines aggregated and processed per day (Ranged from 2008-08-08 to 2015-12-10). It has been scrapped using a scrapping tool downloaded from the Github user [philippremy](https://github.com/philipperemy/Reuters-full-data-set)'s repository. Also, a binary label column refering to Dow Jones index open-close variation (decrease = 0, increase = 1) is provided for each day.

# # Step 1 - Preprocessing Data

# ## Modules
# 
# During this lab, you will work work with modules pandas, scikit-learn, nltk and matplotlib. NLTK (Natural Language Processing Toolkit) is a python module to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

# In[1]:


import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import sklearn
import re


# In[2]:


pandas_version = pd.__version__
numpy_version = np.__version__
nltk_version = nltk.__version__
sklearn_version = sklearn.__version__
re_version = re.__version__
print(pandas_version) # Expect 0.18.1
print(numpy_version) # Expect 1.13.0
print(nltk_version) # Expect 3.2.1
print(sklearn_version) # Expect 0.19.1
print(re_version) # Expect 2.2.1


# ## Some functions
# 
# Below are some pre-written functions that will help you monitor your machine learning pipeline performances.

# In[3]:


from sklearn.feature_extraction.text import CountVectorizer

def check_test(result, expected, display_error):
    """
    Testing your results.
    """
    if result == expected:
        print("1 test passed.")
    else:
        print(display_error)
        
def get_coeffs(model, vect):
    """
    Returns words / n_grams weights in ascending order.
    (param) model : trained scikit learn model.
    (param) vect : used count vectorizer.
    return: word-weight pairwise
    """
    words = vect.get_feature_names()
    coeffs = model.coef_.tolist()[0]
    coeff_df = pd.DataFrame({'word' : words, 
                        'coefficient' : coeffs})
    coeff_df = coeff_df.sort_values(['word', 'coefficient'])
    return coeff_df

def compute_features_frequencies(vect, x_train):
    """
    Computes words frequencies.
    """
    words = vect.get_feature_names()
    frequencies = (x_train.astype(bool).sum(axis=0).astype(float) / x_train.shape[0]).tolist()[0]
    coeff_df = pd.DataFrame({'word' : words, 'frequency' : frequencies})
    return coeff_df


# ## Load data
# 
# Load CSV data as a pandas dataframe and store it in the variable **df**. Get dataframe's number of rows and assign it to the variable **n_rows**. Finally display the first 5 rows with method **.head()**.

# In[4]:


df = pd.read_csv("../data/dow_jones_news.csv", sep=";") # pd.read_csv (sep=";")
n_rows = df.shape[0]
df.head(5)


# In[5]:


check_test(n_rows, 1719, "incorrect number of rows.")


# ## Repartition classes
# 
# For more convenience (and time saving purposes), the data has been preprocessed before the lab. The financial news headers have been aggregated together and separated with \*\*\* pattern. Now, let's take a look on labels distribution. 
# 
# - Assign the number of negative labels to the variable **count_0**
# - Assign the number of positive labels to the variable **count_1**.
# - Assign news of "2012-02-15" to news_20120215.
# - Display all the results.

# In[6]:


count_0 = df[df["Label"] == 0].shape[0]
count_1 = df[df["Label"] == 1].shape[0]
news_20120215 = df[df["Date"] == "2012-02-15"]["news"].iloc[0]
print("count 0 = %s" % count_0)
print("count 1 = %s" % count_1)
print(news_20120215)


# In[7]:


check_test(count_0, 861, "incorrect number of rows.")
check_test(count_1, 858, "incorrect number of rows.")
check_test(news_20120215.split("***")[0], "Study abroad? Why American students head north ", "incorrect news for 2012/02/15")


# # Step 2 - build a first dirty model
# 
# As you have observed, there are plenty of news per day. It is pretty hard for a human to manually predict Dow Jones variations by reading those articles. Let's try to use supervised machine learning. We are going to start with a very simple pipeline.

# ## Train-test split
# 
# Because we don't have a lot of observations to train our machine learning model, we will validate our model with a non-exhaustive K-fold cross-validation procedure. Let start byb splitting our original dataset into 2 subsets with following ratios 70% / 30% :
# - Using function [train_test split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), split the two datasets. Set random state to **203**.
# - Assign resulting train datasets to variables **df_train** and **df_test**.
# - Assign the resulting labels to variables **y_train** and **y_test**.
# - Compute **df_train** and **df_test** number of rows. Assign results to **n_train** and **n_test**.

# In[44]:


from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(df, df["Label"], random_state=203, test_size=0.30)
n_train = y_train.shape[0]
n_test = y_test.shape[0]
print("df_train observations : %s" % n_train)
print("y_train observations : %s" % str(y_train.shape[0]))
print("df_test observations : %s" % n_test)
print("y_test observations : %s" % str(y_test.shape[0]))


# In[45]:


check_test(n_train, 1203, "Err : wrong number of observations")
check_test(n_test, 516, "Err : wrong number of observations")
check_test(y_train.shape[0], 1203, "Err : wrong number of observations")
check_test(y_test.shape[0], 516, "Err : wrong number of observations")


# ## Dealing with text in machine learning
# 
# Text Analysis is a major application field for machine learning algorithms. However one need to proces raw text. Indeed, most of supervised machine learning algortihms expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.
# 
# In order to address this, scikit-learn provides utilities to extract numerical features from text content. A simple approach is :
# - tokenize strings and assign an integer id for each possible token. In other words, we extract words from text (tokens) using white-spaces separators.
# - count the number of occurrences of each token in each document.
# - normalize and weight each token based on token frequency in the documents. For example, a token like "the" is a very frequent word. Therefore, one can decide to weight less this token because it is not relevant for our usecase.
# 
# In this scheme, features and samples are defined as follows:
# - each individual token (word) occurrence frequency (normalized or not) is treated as a feature.
# - the vector of all the token frequencies for a given document is considered as a multivariate sample.
# 
# We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the Bag of Words or “Bag of n-grams” representation. Documents are described by the word occurrences while completely ignoring the relative position information of the words in the document.

# ## Extract simple words features with CountVectorizer

# [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer) converts a collection of text documents to a matrix of token counts. With default configuration, the vectorizer will yield single words tokens delimited by white spaces and punctuation.
# - Create a **CountVectorizer** instance with default parameters. With default setting, the vectorizer handles punctuation and lowercasing.
# - fit the vectorizer on the train dataset and assign the resulting to the variable **vect**. 
# - Once this task is achieved, transform the train set and assign the result to the variable **x_train**.

# In[46]:


from sklearn.feature_extraction.text import CountVectorizer

# Fit and transfrom text with count vectorizer
vectorizer = CountVectorizer()
vect = vectorizer.fit(df_train["news"])
x_train = vect.transform(df_train["news"])


# In[47]:


# Tokenizer example on news
news_test = "Study abroad? Why American students head north *** S.Korea Jan department store sales fall 4.1 pct y/y ***GMP Enhances Merger Value for CVPS Customers ***  Actor Kyle Knies to Star in Cinemax Hit Show The Girl's GuideTo Depravity *** Euro slips on possible delay on Greek bailout *** Integrated Marketing and Branding Expert JimJoseph Joins Cohn & Wolfe to Lead North America Region *** U.S. pushes EU, SWIFT to eject Iran banks *** Dealsof the day -- mergers and acquisitions *** Santaro Receives Government Approval for Launch of First Game *** Moshi Monster maker eyes IPO, U.S. expansion ***"
print(vect.build_tokenizer()(news_test))


# ## How many features do we have ?
# 
# x\_train is now a matrix of N rows and M columns. Each column is related to a specific token (word). How many features do we have? You can answer this question by accessing **vect** attribute ".vocabulary\_". More in documentation : [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer). Assign the result to the variable **m_features**.

# In[48]:


n_features = len(vect.vocabulary_)
print("There are %s features in vocabulary" % n_features)


# In[49]:


check_test(n_features, 191782, "Err : wrong number of features")


# ## Logistic regression
# 
# [Logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) is a simple algorithm that can be used for classification purposes. It models conditional probability P(y=1|x) where x refer to the input features. Contrary to decision trees algorithms (lecture 3), the logistic regression is able to deal with a large number of features. Therefore, it is commonly used for text classification purpose.<br/>
# 
# - Create a simple Logistic Regression instance with default parameters and assign it to variable **clf**.
# - Fit the model with the training data and assign the result to variable **model**.
# - Using the model, make predictions on the train dataset. Asssign the following to **pred_train**.
# - Using clf and [cross_val_predict](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) function, make predictions on the train set in a cross-validation way. Assign the resulting to the variable **pred_val**. (Set number of K-folds K=5).

# In[50]:


from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LogisticRegression

# Fit the model on train dataset
clf = LogisticRegression()
model = clf.fit(x_train, y_train)

# Make predictions on train dataset + validation set (K-fold)
pred_train = model.predict(x_train)
pred_val = cross_val_predict(clf, x_train, y_train, cv=5)


# ## Make evaluation

# It is time to evaluate our first machine learning pipeline. For both train and cross-validation sets, compare predictions you have obtained with real values.
# - Compute [accuracies scores](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) and assign results to variables **train_acc**, **val_acc**.
# - Compute crosstabs using [pandas.crosstab](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html) and assign it to variables **cross_train** and **cross_val**.
# - Display results on standard output.

# In[51]:


from sklearn.metrics import accuracy_score

print("--- train dataset results :")
train_acc = accuracy_score(y_train, pred_train)
cross_train = pd.crosstab(y_train, pred_train)
print(cross_train)
print("Accuracy = %s" % train_acc)

print("\n--- validation set results :")
val_acc = accuracy_score(y_train, pred_val)
cross_val = pd.crosstab(y_train, pred_val)
print(cross_val)
print("Accuracy = %s" % val_acc)


# ## Top and flop words
# 
# Using function get_coeffs() defined at the beginning of this notebook, you can get logistic regression weights associated with every feature (word) :
# - Using this function, get coefficient assigned with every feature. Use variables **model** and **vect** as parameters.
# - Sort resulting dataframe by coefficient importance.
# - Assign the top 40 words to variable **top_40_words** (list format expected).
# - Assign the flop 40 words to variable **flop_40_words** (list format expected).
# - Print all the results.

# In[52]:


df_coeff = get_coeffs(model=model, vect=vect)
df_sort = df_coeff.sort_values(["coefficient", "word"], ascending=False)
top_40_words = df_sort.head(40)["word"].tolist()
flop_40_words = df_sort.tail(40)["word"].tolist()
print(top_40_words)
print(flop_40_words)


# In[53]:


check_test(set(top_40_words), set(['gains', 'rises', 'islamic', 'show', 'hopes', 'general', 'optimism', 'rise', 'healthcare', 'royal', 'data', 'put', 'picks', 'high', 'film', 'opening', 'offers', 'smith', 'moves', 'dow', 'traders', 'stop', 'killed', 'palestinian', 'higher', 'shareholders', 'lift', 'blackrock', 'auction', 'coming', 'gain', 'cost', 'study', 'testing', 'communications', 'workers', 'egypt', 'free', 'join', 'relations']), "Err : wrong top words.")
check_test(set(flop_40_words), set(['bank', 'trust', 'minutes', 'slip', 'banco', 'deutsche', 'election', 'google', 'welcomes', 'jobs', 'game', 'ces', 'weak', 'analytics', 'music', 'get', 'innovation', 'korea', 'august', 'disease', 'african', 'centers', 'chrysler', 'cash', 'blue', 'fears', 'monday', 'bankruptcy', 'solar', 'turkish', 'weigh', 'missing', 'end', 'raises', 'slide', 'low', 'down', 'lower', 'falls', 'worries']), "Err : wrong flop words.")


# ## Questions
# 
# According to you : 
# - Question 1 : what can you say about the results we have obtained on train and validation set?
# - Question 2 : we have obtained a perfect accuracy on train dataset. Why?
# - Question 3 : Is the accuracy metrics well suited for this classification problem? Justify.

# Your answers : 
# _____________________________________________________________________________________________________________________
# _____________________________________________________________________________________________________________________
# _____________________________________________________________________________________________________________________
# _____________________________________________________________________________________________________________________

# # Step 3 - Features engineering

# Features engineering is a main concern for every machine learner. This is the difference between a good machine learning pipeline and a less good one.  It the following, we will process text in order to help the learning and so improve our results.

# ## Remove numbers
# 
# News provided contain a lot of numbers, percents and amounts. CountVectorizer transformer doesn't care about word meaning but only focuses on its writing. With current news, it means that all amounts are assigned to different features and bring noise to our model. Let's remove all the numbers from news.
# - Write a function **"remove_number()"** where input is a string and output is a string where numbers are removed.
# - Apply this function on news for both train and test sets.
# - Once again, train a Countvectorizer to evaluate vocabulary size. Assign the results to **n_features2**.

# In[19]:


import re
def remove_numbers(text):
    return re.sub('[0-9]+', '', text)

df_train["news2"] = df_train["news"].apply(lambda el: remove_numbers(el))
df_test["news2"] = df_test["news"].apply(lambda el: remove_numbers(el))


# In[20]:


vectorizer = CountVectorizer()
vect = vectorizer.fit(df_train["news2"])
n_features2 = len(vect.vocabulary_)
print("There are %s features in vocabulary" % n_features2)


# In[21]:


check_test(remove_numbers("Fitch Withdraws Credit Enhanced Rating on Indiana Finance Authority (IN) Series 2005A-1 and 2008A-1"), "Fitch Withdraws Credit Enhanced Rating on Indiana Finance Authority (IN) Series A- and A-", "Error : wrong remove numbers function")
check_test(remove_numbers("W. R. Berkley Corporation to Announce Third Quarter 2013 Earnings on October 21, 2013"), "W. R. Berkley Corporation to Announce Third Quarter  Earnings on October , ", "Err : wrong remove numbers function")
check_test(n_features2, 180453, "Err : wrong number of features")


# ## Stemming
# 
# In linguistic morphology and information retrieval, stemming is the process for reducing inflected (or sometimes derived) words to their stem, base or root form—generally a written word form. The stem need not be identical to the morphological root of the word. Algorithms for stemming have been studied in computer science since the 1960s. Many search engines treat words with the same stem as synonyms as a kind of query expansion, a process called conflation.
# 
# NLTK module provides several famous stemmers interfaces, such as Porter stemmer, Lancaster Stemmer, Snowball Stemmer etc. In this section, we are going to transform our news using [Porter stemmer](https://tartarus.org/martin/PorterStemmer/).
# - Using the function porter_stemmer() below, apply the stemming on **string_test** and assign the result to the variable **string_test_stem**.
# - Apply the function on column **news2** and return the result to **news3** column (for both train and test sets).
# - Take a look on your stemmed news.
# - Once again, train a Countvectorizer to evaluate vocabulary size. Assign the results to **n_features3**.
# - Want to test Porter stemmer online? See there : http://textanalysisonline.com/nltk-porter-stemmer
# 
# **Note** : In order to accelerate the stemming (which is an heavy process), the function porter_stemmer() is multi-threaded on multiple cores of your computer with the _multiprocessing_ module. However, it can still take some time to run on the full datasets.

# In[22]:


from nltk.stem import PorterStemmer
from multiprocessing import Pool

def porter_stemmer(sentence, cores=4):
    with Pool(processes=cores) as pool:
        stemmer = PorterStemmer()
        result = " ".join(pool.map(stemmer.stem, sentence.split(" ")))
    return result

string_test = "Apple tweaks apps policy under lawmaker pressure *** Rodinia Oil Corp. Announces Statement Concerning Share Price Movement"
string_test_stemmed = porter_stemmer(string_test)
print(string_test_stemmed)


# In[23]:


df_train["news3"] = df_train["news2"].apply(lambda el: porter_stemmer(el))
df_test["news3"] = df_test["news2"].apply(lambda el: porter_stemmer(el))


# In[24]:


vectorizer = CountVectorizer()
vect = vectorizer.fit(df_train["news3"])
n_features3 = len(vect.vocabulary_)
print("There are %s features in vocabulary" % n_features3)


# In[25]:


check_test(porter_stemmer("Fitch Withdraws Credit Enhanced Rating on Indiana Finance Authority (IN) Series 2005A-1 and 2008A-1").lower(), "Fitch Withdraw Credit Enhanc Rate on Indiana Financ Author (IN) Seri 2005A-1 and 2008A-1".lower(), "Error : wrong stemming function")
check_test(porter_stemmer("W. R. Berkley Corporation to Announce Third Quarter 2013 Earnings on October 21, 2013").lower(), "W. R. Berkley Corpor to Announc Third Quarter 2013 Earn on Octob 21, 2013".lower(), "Err : wrong stemming function")
check_test(str(n_features3)[0:-1], "17264", "Err : wrong number of features")


# ## Run a second machine learning algorithm
# 
# During the two previous sections, we have implemented simple simple text processing transformations. Would it yield better results?
# - Once again, train a CountVectorizer and a LogisticRegression with default values on **news3** data.
# - Once again, make predictions on train data and assign the result to **pred_train**.
# - Once again, make predictions on validation set in a K-fold cross-validation way (K=5). assign result to **pred_val**.
# - Evaluate both accuracies and crosstabs. Conclusion?

# In[26]:


from sklearn.feature_extraction.text import CountVectorizer

# Fit and transfrom text with count vectorizer
vectorizer = CountVectorizer()
vect = vectorizer.fit(df_train["news3"])
x_train = vect.transform(df_train["news3"])


# In[27]:


from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LogisticRegression

# Fit the model on train dataset
clf = LogisticRegression()
model = clf.fit(x_train, y_train)

# Make predictions on train dataset + validation set (K-fold)
pred_train = model.predict(x_train)
pred_val = cross_val_predict(clf, x_train, y_train, cv=5)


# In[28]:


from sklearn.metrics import accuracy_score, auc

print("--- train dataset results :")
train_acc = accuracy_score(y_train, pred_train)
cross_train = pd.crosstab(y_train, pred_train)
print(cross_train)
print("Accuracy = %s" % train_acc)

print("\n--- validation set results :")
val_acc = accuracy_score(y_train, pred_val)
cross_val = pd.crosstab(y_train, pred_val)
print(cross_val)
print("Accuracy = %s" % val_acc)


# ## Top and Flop words
# 
# - Once again, assign the top 40 words to variable **top_40_words** (list format expected).
# - Once again, assign the flop 40 words to variable **flop_40_words** (list format expected).
# - Print all results.

# In[29]:


df_coeff = get_coeffs(model=model, vect=vect)
df_sort = df_coeff.sort_values(["coefficient", "word"], ascending=False)
top_40_words = df_sort.head(40)["word"].tolist()
flop_40_words = df_sort.tail(40)["word"].tolist()
print(top_40_words)
print(flop_40_words)


# In[30]:


check_test(set(top_40_words), set(['rise', 'lift', 'gain', 'shoot', 'islam', 'jump', 'hope', 'pick', 'high', 'optim', 'do', 'trader', 'structur', 'show', 'worker', 'healthcare', 'royal', 'dow', 'stop', 'egypt', 'blackrock', 'olymp', 'test', 'come', 'ca', 'respons', 'independ', 'auction', 'panel', 'relat', 'spur', 'spanish', 'bet', 'surg', 'smith', 'sue', 'adopt', 'data', 'pope', 'achiev']), "Err : wrong top words.")
check_test(set(flop_40_words), set(['extend', 'granit', 'end', 'googl', 'concern', 'minut', 'banco', 'sink', 'tumbl', 'asia', 'owner', 'cash', 'august', 'fear', 'chrysler', 'expand', 'histor', 'solar', 'continu', 'suppli', 'monday', 'analyt', 'korea', 'african', 'turkish', 'nfl', 'diseas', 'music', 'th', 'fall', 'blue', 'miss', 'will', 'weigh', 'low', 'slip', 'down', 'lower', 'worri', 'slide']), "Err : wrong flop words.")


# # Step 3 - Features selection
# 
# Using features engineering, we have improved our results but we can still go a bit further. As you have noticed during step 2, our logistic regression is trained on more than 100k features. A lot of these features are not relevant to our model. Features selection can be used to decrease model complexity.

# ## Questions
# 
# According to you : 
# - Question 4 : what kind of error is related to model with a too large number of features?
# - Question 5 : can you list the three features selection approaches we have discussed in lecture 2?
# - Question 6 : which of them seem relevant to you for this text mining use-case?
# 
# Your answers : 
# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

# ## Take a look on features frequencies
# 
# In the following cell, we plot the histogram of document frequencies for each token we have in our vectorizer. A document frequency of 0.5 means that the token (word) appears once every two documents. As you can observe :
# - There are a lot of words (features) that have a very low document frequency.
# - Some words (such as stopwords 'a', 'the', 'some', 'any'...) that have a document frequency close to 1.0. 

# In[31]:


df_freq = compute_features_frequencies(vect, x_train)
plt.figure(figsize=[15, 6])
plt.hist(df_freq["frequency"], bins=200)
plt.xlabel("document frequency")
plt.ylabel("count")
plt.title("Histogram of tokens document frequencies in vectorizer")
plt.show()


# ## Features Selection - Filtering on frequencies
# 
# Filtering features selection is based on a user-defined criteria. A good one in this problem is features frequencies.
# - You can play on both CountVectorizer **min_df** and **max_df** parameters to filter features.
# - Play on these two parameters and observe results you get on both train and validation sets.
# - For each configuration tested, evaluate the number of features you have.
# 
# According to you ?
# - Question 7 : Can you reach a cross-validation score > 0.610 ?
# - Question 8 : what is the justification behind the filtering strategy you have used?
# 
# Your answers : 
# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

# In[32]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_predict

# Fit and transfrom text with count vectorizer and frequencies filtering.
vectorizer = CountVectorizer(min_df=0.05, max_df=0.8)
vect = vectorizer.fit(df_train["news3"])
x_train = vect.transform(df_train["news3"])
n_features = len(vect.vocabulary_)
print("There are %s features in vocabulary" % n_features)

# Fit the model on train dataset
clf = LogisticRegression()
model = clf.fit(x_train, y_train)

# Make predictions on train dataset + validation set (K-fold)
pred_train = model.predict(x_train)
pred_val = cross_val_predict(clf, x_train, y_train, cv=5)

# Results
print("--- train dataset results :")
train_acc = accuracy_score(y_train, pred_train)
cross_train = pd.crosstab(y_train, pred_train)
print(cross_train)
print("Accuracy = %s" % train_acc)

print("\n--- validation set results :")
val_acc = accuracy_score(y_train, pred_val)
cross_val = pd.crosstab(y_train, pred_val)
print(cross_val)
print("Accuracy = %s" % val_acc)


# ## Features selection - Regularization
# 
# Regularization is an embedded features selection approach. It can be used to remove irrelevant features from a cost function point of view. Regularization can be applied through [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). In this step, we will test different regularizations and plot learning curves.
# 
# According to you ?
# - Question 9 : what kind of penalty between **l1** and **l2** will you choose? Justify.
# 
# Your answers : 
# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________
# 
# - Define a range of regularization strength to test in variables **cs**. Keep in mind that the more values you want to test, the more computing time will increase. A power of 10 increasing step is a good way to start but feel free to refine.
# 
# - Through a for loop on **cs** :
# - Train a CountVectorizer with **min_df** and **max_df** with same values than in the previous question.
# - Fit a logistic regression model with regularization.
# - Evaluate accuracies on train and cross-validation dataset. For each step, append accuracies results to lists **train_accs** and **val_accs**.
# - Append the step number to variable **experiments** by increasing the variable **experiment** in the loop.
# - Append the number of non-null regularized features to **non_null_features** list. A way to do so is to access **model coef_**, convert to list and count the number of coefs where the absolute absolute value is greater than 1e-8.
# - Print all relevant information.
# 
# **Note** : Because of high frequencies features filtering we have implemented in previous step, we assume that features have close term frequencies and so features are close in magnitude. Remember that when using regularization, we have to normalize features. on the contrary, some features would dominate and will be highly penalized by regularization. 

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict

train_accs = []
val_accs = []
experiments = []
non_null_features = []
experiment = 0
cs = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

for c in cs:
     # Fit and transfrom text with count vectorizer with frequencies filtering
    vectorizer = CountVectorizer(min_df=0.05, max_df=0.8)
    vect = vectorizer.fit(df_train["news3"])
    x_train = vect.transform(df_train["news3"])
    
    # Fit the model on train dataset
    clf = LogisticRegression(C=c, penalty="l1")
    model = clf.fit(x_train, y_train)
    non_null = len([a for a in model.coef_.tolist()[0] if abs(a) > 0.000000001])

    # Make predictions on train dataset + validation set (K-fold)
    pred_train = model.predict(x_train)
    pred_val = cross_val_predict(clf, x_train, y_train, cv=5)
    train_acc = accuracy_score(y_train, pred_train)
    val_acc = accuracy_score(y_train, pred_val)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    non_null_features.append(non_null)
    experiment += 1
    experiments.append(experiment)
    
    # Print : 
    print("\nexperiment : %s" % experiment)
    print("regularization strength = %s" % c)
    print("accuracy on train dataset : %s" % train_acc)
    print("accuracy on validation dataset : %s" % val_acc)
    print("kept %s / %s features using regularization"% (non_null, len(model.coef_.tolist()[0])))


# ## Plot the learning curves
# 
# Here we are plotting the learning curves.
# 
# According to you :
# - Question 10 : can you identify the underfitting-overfitting tradeoff? With regards to the **non_null_features** evolution, could you justify this learning curve?
# 
# Your answers : 
# 
# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

# In[33]:


plt.figure(figsize=[12,5])
plt.plot(experiments, train_accs, color="b")
plt.plot(experiments, val_accs, color="g")
plt.xlabel("experiment")
plt.ylabel("accuracy")
plt.title("Learning curves - Regularization")
plt.show()


# ## All together on the test set
# 
# During this notebook, we have experimented multiples features engineering and features selection techniques to improve our machine learning model. Time to test the whole pipeline on the test dataset.
# - Train again your best **vect** and **model**, transform and predict the test dataset.
# - Evalute your results.
# 
# According to you :
# - Question 11 : According to you, why does the test set accuracy is a lower than the cross-validation one?
# 
# Your answers : 
# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

# In[34]:


# Fit and transfrom text with count vectorizer
vectorizer = CountVectorizer(min_df=0.05, max_df=0.8)
vect = vectorizer.fit(df_train["news3"])
x_train = vect.transform(df_train["news3"])
x_test = vect.transform(df_test["news3"])
n_features = len(vect.vocabulary_)

# Fit the model on train dataset
clf = LogisticRegression(penalty="l1", C=0.05)
model = clf.fit(x_train, y_train)

# Make predictions on train dataset + validation set (K-fold)
pred_train = model.predict(x_train)
pred_test = model.predict(x_test)
pred_val = cross_val_predict(clf, x_train, y_train, cv=5)

# Results 
print("--- train dataset results :")
train_acc = accuracy_score(y_train, pred_train)
print(pd.crosstab(y_train, pred_train))
print("Accuracy = %s" % train_acc)

print("\n--- validation set results :")
val_acc = accuracy_score(y_train, pred_val)
print(pd.crosstab(y_train, pred_val))
print("Accuracy = %s" % val_acc)

print("\n--- Test set results :")
test_acc = accuracy_score(y_test, pred_test)
print(pd.crosstab(y_test, pred_test))
print("Accuracy = %s" % test_acc)


# # To go further
# 
# During this lab, we have implemented a machine learning pipeline to predict Dow Jones index variations with regards to the news of the day. This work can be the first step to more sophisticated systems (real-time tweets analysis, high-frequencies decisions on the market, forecasting...).
# 
# Here are some ideas if you want to go futher on this lab : 
# 
# - **Using TfIdf vectorizer** : in a large text corpus, some words will be very present (e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information about the actual contents of the document. When the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms. In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the [tf–idf transform](http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting).<br/><br/>
# 
# - **Using LinearSVC classifier** : logistic regression we have implemented tries to minimize cross-entropy cost-function ("error" in classification). [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) a simple Support Vector Machine implementation uses an other cost function named Hinge Loss. This metrics measures the "margin" between the classifier and closest observations. It does not use full data to learn. This algorithm is often less sensitive to high dimensionality than Logistic regression. Moreover, it can be setted in a very simple way using only "C" hyperparameter. More information in this [quora thread](https://www.quora.com/Machine-Learning-How-are-logistic-regression-and-linear-SVMs-similiar).
# 
# <img src="../utils/linearsvc.png" alt="Master 203", width=200>
# 
# - [**Finally, here is a paper**](http://emnlp2014.org/papers/pdf/EMNLP2014148.pdf) from Harbin Institute of technology in China about S&P500 variations predictions using machine learning. Team uses more advanced news modeling and neural nets (lecture 4) models but the methodology is really close to our lab. The team is able to reach about 60% accuracy on S&P500 predictions index and about 70% on individual stocks. Really interesting !

# In[ ]:




