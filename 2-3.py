#!/usr/bin/env python
# coding: utf-8

# <img src="../utils/title.png" alt="title", width="600">

# <img src="../utils/master_203_pic.jpg" alt="Master 203", width=400>

# ## About Sberbank russian housing market challenge

# In 2017, Sberbank Russia’s oldest and largest bank has challenged data scientists from all around the world with a machine learning competiton ($25,000 prizes). In this competition, competitors (kagglers) had to develop algorithms which use a broad spectrum of features to predict russian housing market prices. Competitors rely on a rich dataset that includes housing data and macroeconomic patterns. Winning models have allowed Sberbank to provide more certainty to their customers in an uncertain economy. in this live-coding, we will experiment different agorithms and supervised machine learning pipeline features.
# 
# See more on : [Kaggle competition web page](https://www.kaggle.com/c/sberbank-russian-housing-market)

# # Lecture 3 - Decision Trees
# 
# In this part, we will try improve our performance using tree-based methods. Used features will be the same that for lecture 2 live coding in order to compare models. Lecture 2 Live coding achieved performance was :
# - RMSLE train = 0.4813
# - RMSLE validation = 0.5045
# - RMSLE test = 0.5022

# In[3]:


import pandas as pd
print("\n----- Loading data")
df = pd.read_csv("../data/train.csv")
df_macro = pd.read_csv("../data/macro.csv")
df = df.merge(df_macro, how="left", on="timestamp")
print("dataframe shape : %s" % str(df.shape))
print("dataframe macro shape : %s" % str(df_macro.shape))
print("Merging df and df_macro..")

for col in df.columns:
    print("Column %s : Number of NaN = %s" % (col, df[col].isnull().sum()))


# In[5]:


original_features = ["full_sq", "life_sq", "floor", "product_type", "sub_area", "area_m", "green_zone_part", "indust_part",
           "children_preschool", "indust_part", "children_school", "healthcare_centers_raion", "university_top_20_raion", "metro_km_avto",
           "metro_min_avto", "kindergarten_km", "park_km", "kremlin_km", "cafe_count_500", "cpi", "ppi",
           "gdp_deflator", "balance_trade", "usdrub", "eurrub", "brent", "net_capital_export",
           "gdp_annual", "gdp_annual_growth", "average_provision_of_build_contract", "rts",
           "micex", "mortgage_value", "deposits_value", "salary", "salary_growth", "unemployment"]


# ## Setup : run lecture 2 processing pipeline

# In[6]:


# Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from math import sqrt
from math import log

def compute_rmsle(y_true, y_pred):
    return sqrt(mean_squared_log_error(y_true, y_pred))

def one_hot_encoder(df, col_to_encode, col_is_int=True):
    """
    Performs One Hot Encoding with sklearn.
    df : pandas dataframe containing column to encode
    col_to_encode : (str) column name
    col_is_int : (bool) whether or not column is integer.
    """
    # Create the mapping
    if col_is_int: 
        integer_encoded = df[col_to_encode].reshape(-1, 1)
    else:
        values = df[col_to_encode].tolist()
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
    
    # One hot encoder
    ohe = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    df_temp = pd.DataFrame(ohe.fit_transform(integer_encoded), index=df.index)
    df_temp.columns = ["%s_%s" % (col_to_encode, col) for col in df_temp.columns]
    df_results = pd.concat([df, df_temp], axis=1)
    return df_results

# Loading dataset
print("\n----- Loading data")
df = pd.read_csv("../data/train.csv")
df_macro = pd.read_csv("../data/macro.csv")
df = df.merge(df_macro, how="left", on="timestamp")
print("dataframe shape : %s" % str(df.shape))
print("dataframe macro shape : %s" % str(df_macro.shape))
print("Merging df and df_macro..")

# Select and clean features
print("\n----- Select and clean features :")
for col in ["state", "life_sq", "floor", "num_room"]:
    df[col] = df[col].fillna(df[col].median())
    print("Column %s : Number of NaN = %s" % (col, df[col].isnull().sum()))

# Features engineering
print("\n----- Features engineering :")
print("Binarizing column 'product_investment'")
df["product_investment"] = df["product_type"].apply(lambda el: 1.0 if el == "Investment" else 0)
print("One-hot encoding column 'sub_area'")
df = one_hot_encoder(df, "sub_area", False)
print("One-hot encoding column 'state'")
df["state"] = df["state"].apply(lambda el: 2.0 if el >=5 else el)
df = one_hot_encoder(df, "state", True)
print("Rescale price doc")
df["price_doc_scale"] = df["price_doc"].apply(lambda el: log(el)) 

# Train-Val-Test split
print("\n----- Train-Validation-Test split")
features = original_features + [col for col in df.columns if "state_" in col] + [col for col in df.columns if "sub_area_" in col]
for f in ["sub_area", "product_type"]:
    features.remove(f)
x_train_val, x_test, y_train_val, y_test = train_test_split(df[features], df["price_doc_scale"], test_size=0.2, random_state=203)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=203)
print("Train dataset shape : %s" % str(x_train.shape))
print("validation dataset shape : %s" % str(x_val.shape))
print("Test dataset shape : %s" % str(x_test.shape))

# Display
pd.set_option("display.max_columns", 500)
x_train.head(10)


# ## Run a simple decision tree with max_depth = 5
# 
# Let's run a vanilla decision tree with max_depth = 5.

# In[117]:


from sklearn.tree import DecisionTreeRegressor
import math

# Fit a simple linear regression model
clf = DecisionTreeRegressor(max_depth=5)
model = clf.fit(x_train, y_train)

# Make prediction on train, validation and test set.
pred_train = pd.Series(model.predict(x_train), index=y_train.index)
pred_val = pd.Series(model.predict(x_val), index=y_val.index)
pred_test = pd.Series(model.predict(x_test), index=y_test.index)

# Rescale price_docs and predictions
pred_train = pred_train.apply(lambda el: math.exp(el))
pred_val = pred_val.apply(lambda el: math.exp(el))
pred_test = pred_test.apply(lambda el: math.exp(el))
y_train_exp = y_train.apply(lambda el: math.exp(el))
y_val_exp = y_val.apply(lambda el: math.exp(el))
y_test_exp = y_test.apply(lambda el: math.exp(el))

# Compute MSLE evaluation metrics
rmsle_train = compute_rmsle(y_train_exp, pred_train)
rmsle_val = compute_rmsle(y_val_exp, pred_val)
rmsle_test = compute_rmsle(y_test_exp, pred_test)
print("RMSLE train : %s" % rmsle_train)
print("RMSLE validation : %s" % rmsle_val)
print("RMSLE test : %s" % rmsle_test)


# ## Learning curves - play on tree max_depth 
# 
# Diagnose tree depth impact on underfitting-overfitting tradeoff using learning curves.

# In[118]:


from sklearn.tree import DecisionTreeRegressor
import math

# Fit a simple linear regression model
depths = range(1, 25)
all_rmsle_train = []
all_rmsle_val = []
experiments  = []

for depth in depths: 
    clf = DecisionTreeRegressor(max_depth=depth)
    model = clf.fit(x_train, y_train)

    # Make prediction on train, validation and test set.
    pred_train = pd.Series(model.predict(x_train), index=y_train.index)
    pred_val = pd.Series(model.predict(x_val), index=y_val.index)

    # Rescale price_docs and predictions
    pred_train = pred_train.apply(lambda el: math.exp(el))
    pred_val = pred_val.apply(lambda el: math.exp(el))
    y_train_exp = y_train.apply(lambda el: math.exp(el))
    y_val_exp = y_val.apply(lambda el: math.exp(el))

    # Compute MSLE evaluation metrics
    rmsle_train = compute_rmsle(y_train_exp, pred_train)
    rmsle_val = compute_rmsle(y_val_exp, pred_val)
    all_rmsle_train.append(rmsle_train)
    all_rmsle_val.append(rmsle_val)
    print("depth = %s | RMSLE train = %s | RMSLE val = %s " % (depth, rmsle_train, rmsle_val))
    
plt.plot(depths, all_rmsle_train, color="b")
plt.plot(depths, all_rmsle_val, color="g")
plt.xlabel("tree depth")
plt.ylabel("RMSLE")
plt.title("Learning curves - playing on tree depth")
plt.show()


# ## Prevent overfitting with min_samples_split

# In[127]:


# Fit a simple linear regression model
depths = range(1, 20)
min_samples = [2, 6, 10, 15, 20, 30, 50, 70]
all_rmsle_train = []
all_rmsle_val = []

for min_s in min_samples:
    temp_rmsle_train = []
    temp_rmsle_val = []
    min_rmsle_val = 10
    best_conf = 0
    print("Compute trees with min_samples_splits = %s" % min_s)
    for depth in depths: 
        clf = DecisionTreeRegressor(max_depth=depth, min_samples_split=min_s)
        model = clf.fit(x_train, y_train)

        # Make prediction on train, validation and test set.
        pred_train = pd.Series(model.predict(x_train), index=y_train.index)
        pred_val = pd.Series(model.predict(x_val), index=y_val.index)

        # Rescale price_docs and predictions
        pred_train = pred_train.apply(lambda el: math.exp(el))
        pred_val = pred_val.apply(lambda el: math.exp(el))
        y_train_exp = y_train.apply(lambda el: math.exp(el))
        y_val_exp = y_val.apply(lambda el: math.exp(el))

        # Compute MSLE evaluation metrics
        rmsle_train = compute_rmsle(y_train_exp, pred_train)
        rmsle_val = compute_rmsle(y_val_exp, pred_val)
        temp_rmsle_train.append(rmsle_val)
        temp_rmsle_val.append(rmsle_val)
        
        # Checking minimum
        if rmsle_val < min_rmsle_val:
            min_rmsle_val = rmsle_val
            best_conf = depth
    print("--> Min RMSLE on validation set = %s for depth = %s" % (min_rmsle_val, best_conf))
    all_rmsle_train.append(temp_rmsle_train)
    all_rmsle_val.append(temp_rmsle_val)


# In[128]:


# Make plots
for (curve, min_sample) in zip(all_rmsle_val, min_samples):
    plt.plot(depths, curve, label="min samples = %s" % min_sample)
plt.xlabel("tree depth")
plt.ylabel("RMSLE")
plt.legend(loc="upper left", prop={'size': 8})
plt.title("Min samples splits impact on validation set")
plt.show()

for (curve, min_sample) in zip(all_rmsle_val, min_samples):
    plt.plot(depths[4:9], curve[4:9], label="min samples = %s" % min_sample)
plt.xlabel("tree depth")
plt.ylabel("RMSLE")
plt.legend(loc="upper left", prop={'size': 8})
plt.title("Min samples splits impact on validation set - zoom")
plt.show()


# ## Prevent overfitting with min_samples_leaf

# In[129]:


# Fit a simple linear regression model
depths = range(1, 20)
min_leaf = [1, 10, 30, 50, 100, 300, 1000]
all_rmsle_train = []
all_rmsle_val = []

for min_l in min_leaf:
    temp_rmsle_train = []
    temp_rmsle_val = []
    min_rmsle_val = 10
    best_conf = 0
    print("Compute trees with min_samples_leaf = %s" % min_l)
    for depth in depths: 
        clf = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_l)
        model = clf.fit(x_train, y_train)

        # Make prediction on train, validation and test set.
        pred_train = pd.Series(model.predict(x_train), index=y_train.index)
        pred_val = pd.Series(model.predict(x_val), index=y_val.index)

        # Rescale price_docs and predictions
        pred_train = pred_train.apply(lambda el: math.exp(el))
        pred_val = pred_val.apply(lambda el: math.exp(el))
        y_train_exp = y_train.apply(lambda el: math.exp(el))
        y_val_exp = y_val.apply(lambda el: math.exp(el))

        # Compute MSLE evaluation metrics
        rmsle_train = compute_rmsle(y_train_exp, pred_train)
        rmsle_val = compute_rmsle(y_val_exp, pred_val)
        temp_rmsle_train.append(rmsle_val)
        temp_rmsle_val.append(rmsle_val)
        
        # Checking minimum
        if rmsle_val < min_rmsle_val:
            min_rmsle_val = rmsle_val
            best_conf = depth
    print("--> Min RMSLE on validation set = %s for depth = %s" % (min_rmsle_val, best_conf))
    all_rmsle_train.append(temp_rmsle_train)
    all_rmsle_val.append(temp_rmsle_val)


# In[130]:


# Make plots
for (curve, min_l) in zip(all_rmsle_val, min_leaf):
    plt.plot(depths, curve, label="min samples per leaf = %s" % min_l)
plt.xlabel("tree depth")
plt.ylabel("RMSLE")
plt.legend(loc="upper left", prop={'size': 8})
plt.title("Min samples splits impact on validation set")
plt.show()

for (curve, min_l) in zip(all_rmsle_val, min_leaf):
    plt.plot(depths[4:15], curve[4:15], label="min samples per leaf = %s" % min_l)
plt.xlabel("tree depth")
plt.ylabel("RMSLE")
plt.legend(loc="upper left", prop={'size': 8})
plt.title("Min samples splits impact on validation set - zoom")
plt.show()


# ## Run best model on test set

# In[131]:


# Best model
clf = DecisionTreeRegressor(max_depth=10, min_samples_leaf=100)
model = clf.fit(x_train, y_train)

# Make prediction on train, validation and test set.
pred_train = pd.Series(model.predict(x_train), index=y_train.index)
pred_val = pd.Series(model.predict(x_val), index=y_val.index)
pred_test = pd.Series(model.predict(x_test), index=y_test.index)

# Rescale price_docs and predictions
pred_train = pred_train.apply(lambda el: math.exp(el))
pred_val = pred_val.apply(lambda el: math.exp(el))
pred_test = pred_test.apply(lambda el: math.exp(el))
y_train_exp = y_train.apply(lambda el: math.exp(el))
y_val_exp = y_val.apply(lambda el: math.exp(el))
y_test_exp = y_test.apply(lambda el: math.exp(el))

# Compute MSLE evaluation metrics
rmsle_train = compute_rmsle(y_train_exp, pred_train)
rmsle_val = compute_rmsle(y_val_exp, pred_val)
rmsle_test = compute_rmsle(y_test_exp, pred_test)
print("-- RMSLE train : %s" % rmsle_train)
print("-- RMSLE validation : %s" % rmsle_val)  
print("-- RMSLE test : %s" % rmsle_test)


# # Lecture 3 - Random Forests
# 
# Random forest is an aggregation of low-corelated trees to reduce variance and so adress overfitting situations. It is based on both bagging + features sampling principles to build low-correlated trees.

# ## Compare decision trees with random Forests
# 
# Let us compare simple decision tree and random forest with no tree learning stopping (min_samples_split / min_samples_leaf with default values). We will run the two following models for different max depths.
# - Decision tree regressor.
# - Random Forest regressor (n_estimators = 100, n_jobs=-1).

# In[7]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import math

# Prerequisites :
depths = range(1, 18)
all_rmsle_train = []
all_rmsle_val = []
experiments  = []
all_rmsle_train_tree = []
all_rmsle_train_forest = []
all_rmsle_val_tree = []
all_rmsle_val_forest = []


for depth in depths: 
    # Decision tree
    clf_tree = DecisionTreeRegressor(max_depth=depth)
    model_tree = clf_tree.fit(x_train, y_train)
    
    # Random Forest
    clf_forest = RandomForestRegressor(n_estimators=100, max_depth=depth, n_jobs=-1)
    model_forest = clf_forest.fit(x_train, y_train)
    
    for model, all_rmsle_train, all_rmsle_val, typ in zip([model_tree, model_forest], [all_rmsle_train_tree, all_rmsle_train_forest],  [all_rmsle_val_tree, all_rmsle_val_forest] ,["tree", "forest"]):
        # Make prediction on train, validation and test set.
        pred_train = pd.Series(model.predict(x_train), index=y_train.index)
        pred_val = pd.Series(model.predict(x_val), index=y_val.index)

        # Rescale price_docs and predictions
        pred_train = pred_train.apply(lambda el: math.exp(el))
        pred_val = pred_val.apply(lambda el: math.exp(el))
        y_train_exp = y_train.apply(lambda el: math.exp(el))
        y_val_exp = y_val.apply(lambda el: math.exp(el))

        # Compute MSLE evaluation metrics
        rmsle_train = compute_rmsle(y_train_exp, pred_train)
        rmsle_val = compute_rmsle(y_val_exp, pred_val)
        all_rmsle_train.append(rmsle_train)
        all_rmsle_val.append(rmsle_val)
        print("depth = %s | RMSLE train = %s | RMSLE val = %s (%s)" % (depth, rmsle_train, rmsle_val, typ))


# In[85]:


plt.plot(depths, all_rmsle_train_tree, color="b", label="Decision tree - train")
plt.plot(depths, all_rmsle_val_tree, color="g", label="Decision tree - validation")
plt.plot(depths, all_rmsle_train_forest, color="y", label="Random Forest - train")
plt.plot(depths, all_rmsle_val_forest, color="r", label="Random Forest - validation")                                                  
plt.xlabel("tree depth")
plt.ylabel("RMSLE")
plt.title("Learning curves - playing on tree depth")
plt.legend(loc="lower left", prop={'size': 8})
plt.show()


# ## Take a look on features importance
# 
# We are currently using more than 180 features in our decision trees / random forests algorithms. This large number of features yields bad consequences : 
# - Overfitting in the bottom nodes.
# - Trees learning latencies.
# 
# In order to remove some features (features selection), we will take a look on features importances. For a sklearn tree, a feature is considered important if : 
# - it is used frequently for splits in decision tree.
# - splits are used high up in the tree.

# In[8]:


# Run a random forest :
clf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1)
forest = clf.fit(x_train, y_train)

# Compute features importances
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 50 features
print("plot top 50 features importance :")
plt.figure(figsize=[15,6])
plt.title("Feature importances")
plt.bar(range(0, 50), importances[indices][:50].tolist(), color="r", align="center")
plt.xticks(range(0, 50), x_train.columns[indices[:50]], rotation=90)
plt.xlim([-1, 51])
plt.show()

# Display full features ranking : 
print("Feature ranking:")
for f in range(x_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, x_train.columns[indices[f]], importances[indices[f]]))


# ## Reduce overfitting - All together
# 
# In this last part, we will adress overfitting problem with multiple solutions we have : 
# - Filter dataset on top 30 features (features selection, also more convenient with latencies).
# - Run Random Forest algorithm (n_estimators = 300, max_features between 10 and 20).
# - min_samples_leaf between 50 and 100.
# - min_samples_split between 50 and 70.
# - max_depth is equal to 12.
# 
# We will search for the best combination using greedy Search. Some sklearn objects are very convenient when searching for best set of hyperparameters : 
# - ParameterGrid to create a set of all possible user-defined hyperparameters (see more in the doc : [ParameterGrid](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html)
# - GridSearchCV to perform grid search in a cross-validation way (see more in the doc : [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

# In[9]:


from sklearn.model_selection import ParameterGrid

# Define a grid object :
param_grid = {'max_depth': [12], 'n_estimators': [300], "max_features": [10, 15, 20], 
              "min_samples_leaf": [50, 70, 100], "min_samples_split": [50, 70], "n_jobs": [-1]}

# For each conf, run the model and evaluate it.
all_rmsle_train = []
all_rmsle_val = []
experiments = []
i = 1
x_train_features = x_train[x_train.columns[indices[:50]]]
x_val_features = x_val[x_train.columns[indices[:50]]]
x_test_features = x_test[x_train.columns[indices[:50]]]

for conf in ParameterGrid(param_grid):
    # fit model and make predictions :
    clf = RandomForestRegressor(**conf)
    model = clf.fit(x_train_features, y_train)
    pred_train = pd.Series(model.predict(x_train_features), index=y_train.index)
    pred_val = pd.Series(model.predict(x_val_features), index=y_val.index)

    # Rescale price_docs and predictions
    pred_train = pred_train.apply(lambda el: math.exp(el))
    pred_val = pred_val.apply(lambda el: math.exp(el))
    y_train_exp = y_train.apply(lambda el: math.exp(el))
    y_val_exp = y_val.apply(lambda el: math.exp(el))

    # Compute RMSLE evaluation metrics
    rmsle_train = compute_rmsle(y_train_exp, pred_train)
    rmsle_val = compute_rmsle(y_val_exp, pred_val)
    all_rmsle_train.append(rmsle_train)
    all_rmsle_val.append(rmsle_val)
    experiments.append(i)
    print("\n----- Experiment %s :" % i)
    print(conf)
    print("--> RSMLE train = %s" % rmsle_train)
    print("--> RSMLE val = %s" % rmsle_val)
    i +=1

# Get best model results on test set :
print("\n----- Best configuration -----")
best_model_index = all_rmsle_val.index(min(all_rmsle_val))
print("best experiment = %s" % experiments[best_model_index])
print(ParameterGrid(param_grid)[best_model_index])
print("--> Best RSMLE train : %s" % all_rmsle_train[best_model_index])
print("--> Best RSMLE val : %s" % all_rmsle_val[best_model_index])
print("Run best config on test set..")
clf = RandomForestRegressor(**ParameterGrid(param_grid)[best_model_index])
model = clf.fit(x_train_features, y_train)
pred_test = pd.Series(model.predict(x_test_features), index=y_test.index)
pred_test = pred_test.apply(lambda el: math.exp(el))
y_test_exp = y_test.apply(lambda el: math.exp(el))
rmsle_test = compute_rmsle(y_test_exp, pred_test)
print("--> Best RSMLE test : %s" % rmsle_test)

