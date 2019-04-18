#!/usr/bin/env python
# coding: utf-8

# <img src="../utils/title.png" alt="title", width="600">

# <img src="../utils/master_203_pic.jpg" alt="Master 203", width=400>

# ## About Sberbank russian housing market challenge

# In 2017, Sberbank Russia’s oldest and largest bank has challenged data scientists from all around the world with a machine learning competiton ($25,000 prizes). In this competition, competitors (kagglers) had to develop algorithms which use a broad spectrum of features to predict russian housing market prices. Competitors rely on a rich dataset that includes housing data and macroeconomic patterns. Winning models have allowed Sberbank to provide more certainty to their customers in an uncertain economy. in this live-coding, we will experiment different agorithms and supervised machine learning pipeline features.
# 
# See more on : [Kaggle competition web page](https://www.kaggle.com/c/sberbank-russian-housing-market)

# # Lecture 2 - Model Evaluation
# 
# In this part, we will implement model evaluation through a train-validation-test split. We will also introduce RMSLE metrics (Root Mean Square Logarithmic Error) to evaluate our model performance. Finally, we will run learning curves to diagnose underfittin-ovserffiting tradeoff with respect to polynom order.

# ## Modules

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Load train and test datasets

# In[16]:


# Loading dataset
df = pd.read_csv("../data/train.csv")
df = df.dropna(subset=["full_sq", "life_sq", "floor", "num_room"]) # We drop NA until lecture on data cleaning.
print("dataframe shape : %s" % str(df.shape))


# ## Train-validation-test split

# In[17]:


from sklearn.model_selection import train_test_split

# Train-val-test split validation
x_train_val, x_test, y_train_val, y_test = train_test_split(df[["full_sq", "life_sq", "num_room"]], df["price_doc"], test_size=0.2, random_state=203)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=203)
print("Train dataset shape : %s" % x_train.shape[0])
print("validation dataset shape : %s" % x_val.shape[0])
print("Test dataset shape : %s" % x_test.shape[0])


# ## Evaluation metrics : RMSLE

# In[18]:


from sklearn.metrics import mean_squared_log_error
from math import sqrt

def compute_rmsle(y_true, y_pred):
    return sqrt(mean_squared_log_error(y_true, y_pred))


# ## Compute RMSLE for simple linear regression

# In[19]:


from sklearn.linear_model import LinearRegression

# Fit a simple linear regression model
clf = LinearRegression()
model = clf.fit(x_train, y_train)

# Make prediction on train, validation and test set.
pred_train = pd.Series(model.predict(x_train), index=y_train.index).apply(lambda el: 1e-6 if el < 0 else el)
pred_val = pd.Series(model.predict(x_val), index=y_val.index).apply(lambda el: 1e-6 if el < 0 else el)
pred_test = pd.Series(model.predict(x_test), index=y_test.index).apply(lambda el: 1e-6 if el < 0 else el)

# Compute MSLE evaluation metrics
rmsle_train = compute_rmsle(y_train, pred_train)
rmsle_val = compute_rmsle(y_val, pred_val)
rmsle_test = compute_rmsle(y_test, pred_test)
print("RMSLE train : %s" % rmsle_train)
print("RMSLE validation : %s" % rmsle_val)
print("RMSLE test : %s" % rmsle_test)


# ## Learning curves - What is optimal polynom order ?

# In[20]:


from sklearn.preprocessing import PolynomialFeatures, StandardScaler

all_rmsle_train = []
all_rmsle_val = []
experiments = []
poly_range = range(1, 8)
i = 0

for poly in poly_range:
    
    # Create polynomial features
    polynomial_order = poly
    poly = PolynomialFeatures(polynomial_order)
    x_train_features = poly.fit_transform(x_train)
    x_val_features = poly.fit_transform(x_val)
    
    # Standardize features using sklearn Standardscaler
    scaler = StandardScaler()
    scaler = scaler.fit(x_train_features)
    x_train_features = scaler.transform(x_train_features)
    x_val_features = scaler.transform(x_val_features)
 
    # Train a simple linear regression algorithm
    clf = LinearRegression()
    model = clf.fit(x_train_features, y_train)
    
    # Make prediction on train, validation and test set.
    pred_train = pd.Series(model.predict(x_train_features), index=y_train.index).apply(lambda el: 1e-6 if el < 0 else el)
    pred_val = pd.Series(model.predict(x_val_features), index=y_val.index).apply(lambda el: 1e-6 if el < 0 else el)
    
    # Compute MSLE evaluation metrics
    rmsle_train = compute_rmsle(y_train, pred_train)
    rmsle_val = compute_rmsle(y_val, pred_val)
    all_rmsle_train.append(rmsle_train)
    all_rmsle_val.append(rmsle_val)
    experiments.append(i)
    i += 1

plt.plot(poly_range, all_rmsle_train)
plt.plot(poly_range, all_rmsle_val, color="g")
plt.xlabel("Experiments (increasing polynomial order)")
plt.ylabel(("Root Mean Square Logarithm Error"))
plt.title("Learning Curves -  polynom order")
plt.show()


# ## Testing best model

# In[21]:


from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    
# Create polynomial features
polynomial_order = 2
poly = PolynomialFeatures(polynomial_order)
x_train_features = poly.fit_transform(x_train)
x_val_features = poly.fit_transform(x_val)
x_test_features = poly.fit_transform(x_test)

# Standardize features using sklearn Standardscaler
scaler = StandardScaler()
scaler = scaler.fit(x_train_features)
x_train_features = scaler.transform(x_train_features)
x_val_features = scaler.transform(x_val_features)
x_test_features = scaler.transform(x_test_features)

# Train a simple linear regression algorithm
clf = LinearRegression()
model = clf.fit(x_train_features, y_train)
    
# Make prediction on train, validation and test set.
pred_train = pd.Series(model.predict(x_train_features), index=y_train.index).apply(lambda el: 1e-6 if el < 0 else el)
pred_val = pd.Series(model.predict(x_val_features), index=y_val.index).apply(lambda el: 1e-6 if el < 0 else el)
pred_test = pd.Series(model.predict(x_test_features), index=y_test.index).apply(lambda el: 1e-6 if el < 0 else el)

# Compute MSLE evaluation metrics
rmsle_train = compute_rmsle(y_train, pred_train)
rmsle_val = compute_rmsle(y_val, pred_val)
rmsle_test = compute_rmsle(y_test, pred_test)
print("RMSLE train : %s" % rmsle_train)
print("RMSLE validation : %s" % rmsle_val)
print("RMSLE test : %s" % rmsle_test)


# # Lecture 2 - Features engineering and selection

# ## Modules

# In[77]:


import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


# ## Reloading data

# In[78]:


# Loading dataset
df = pd.read_csv("../data/train.csv")
df_macro = pd.read_csv("../data/macro.csv")
print("dataframe shape : %s" % str(df.shape))
print("dataframe macro shape : %s" % str(df_macro.shape))


# ## Display Macro dataframe

# In[79]:


pd.set_option("display.max_columns", 500)
df_macro.head()


# ## Merge df and df_macro on timestamp

# In[80]:


df = df.merge(df_macro, how="left", on="timestamp")
print("dataframe shape : %s" % str(df.shape))


# ## Select a set of relevant features

# In[81]:


original_features = ["full_sq", "state", "product_type", "kremlin_km", "cpi", "usdrub", "eurrub", "unemployment", "floor", "num_room", "sub_area"]
for col in original_features:
    print("Column %s : Number of NaN = %s" % (col, df[col].isnull().sum()))


# ## Clean dataset with median values

# In[82]:


for col in ["state", "floor", "num_room"]:
    df[col] = df[col].fillna(df[col].median())
    
for col in original_features:
    print("Column %s : Number of NaN = %s" % (col, df[col].isnull().sum()))


# ## Binarize product_type

# In[84]:


# Binarize variable product_type
df["product_investment"] = df["product_type"].apply(lambda el: 1.0 if el == "Investment" else 0)
print(df["product_investment"].value_counts())


# ## One Hot Encode feature sub_area

# In[85]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Map sub_area -> index
sub_areas = df["sub_area"].tolist()
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(sub_areas)
print("Mapping sub_area to index :")
print(integer_encoded)

# Map index -> new column using ONe Hot encoder
ohe = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
df_sub_areas = pd.DataFrame(ohe.fit_transform(integer_encoded), index=df.index)
df_sub_areas.columns = ["sub_area_%s" % col for col in df_sub_areas.columns]
print("\nResults of one hot encoding : ")
df_sub_areas.head(5)

# Concatenate sub_areas with original dataframe :
df = pd.concat([df, df_sub_areas], axis=1)
df.head(5)


# ## Take a look on variable state

# In[86]:


plt.hist(df["state"], bins=100)
plt.show()


# ## Clean variable state and one-hot encode it

# In[87]:


# Clean outliers
df["state"] = df["state"].apply(lambda el: 2.0 if el >=5 else el)

# Ont Hot Encoding
ohe = OneHotEncoder(sparse=False)
integer_encoded = df["state"].reshape(-1, 1)
df_state = pd.DataFrame(ohe.fit_transform(integer_encoded), index=df.index)
df_state.columns = ["state_%s" % col for col in df_state.columns]
print("\nResults of one hot encoding : ")
df_sub_areas.head(5)

# Concatenate sub_areas with original dataframe :
df = pd.concat([df, df_state], axis=1)
df.head(5)


# ## Rescale column price_doc
# 
# Because evaluation metrics is RMSLE (Root Mean Square Log Error), it is better to scale price_doc column with logarithm scale. 

# In[97]:


from math import log
df["price_doc_scale"] = df["price_doc"].apply(lambda el: log(el)) 


# ## Train-Validation-Test split

# In[108]:


from sklearn.model_selection import train_test_split

# Select features
features =  ["full_sq", "product_investment", "kremlin_km", "cpi", "usdrub", "eurrub", "unemployment", "floor", "num_room"] 
features = features + [col for col in df.columns if "state_" in col] #+ [col for col in df.columns if "sub_area_" in col]

# Train-val-test split validation
x_train_val, x_test, y_train_val, y_test = train_test_split(df[features], df["price_doc_scale"], test_size=0.2, random_state=203)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=203)
print("Train dataset shape : %s" % x_train.shape[0])
print("validation dataset shape : %s" % x_val.shape[0])
print("Test dataset shape : %s" % x_test.shape[0])


# ## Train simple linear regression

# In[109]:


from sklearn.linear_model import LinearRegression
import math

# Fit a simple linear regression model
clf = LinearRegression()
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


# ## Polynomial regression and regularization

# In[115]:


from sklearn.linear_model import Lasso
import math

# Create polynomial features (polynomial order = 3)
polynomial_order = 3
poly = PolynomialFeatures(polynomial_order, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_val_poly = poly.fit_transform(x_val)
x_test_poly = poly.fit_transform(x_test)
print("x_train_poly shape : %s" % str(x_train_poly.shape))

# Standardize features using sklearn Standardscaler
scaler = StandardScaler()
scaler = scaler.fit(x_train_poly)
x_train_poly = scaler.transform(x_train_poly)
x_val_poly = scaler.transform(x_val_poly)
x_test_poly = scaler.transform(x_test_poly)

lambda_range = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
all_rmsle_train = []
all_rmsle_val = []
experiments  = []
i = 1
for lamb in lambda_range:
    # Train a simple linear regression algorithm
    print("\nLasso regularization strength = %s" % lamb)
    clf = Lasso(alpha=lamb)
    model = clf.fit(x_train_poly, y_train)

    # Make prediction on train and validation sets.
    pred_train = pd.Series(model.predict(x_train_poly), index=y_train.index).apply(lambda el: 1e-6 if el < 0 else el)
    pred_val = pd.Series(model.predict(x_val_poly), index=y_val.index).apply(lambda el: 1e-6 if el < 0 else el)
    
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
    experiments.append(i)
    i += 1
    print("-- RMSLE train : %s" % rmsle_train)
    print("-- RMSLE validation : %s" % rmsle_val)


# ## Plot learning curves

# In[118]:


plt.plot(experiments, all_rmsle_train, color="b")
plt.plot(experiments, all_rmsle_val, color="g")
plt.xlabel("experiment")
plt.ylabel("RMSLE")
plt.title("Learning curves - playing on Lasso regularization strength")
plt.show()


# ## Testing best model on test set

# In[119]:


# Best model
clf = Lasso(alpha=0.0005)
model = clf.fit(x_train_poly, y_train)

# Make prediction on train, validation and test set.
pred_train = pd.Series(model.predict(x_train_poly), index=y_train.index).apply(lambda el: 1e-6 if el < 0 else el)
pred_val = pd.Series(model.predict(x_val_poly), index=y_val.index).apply(lambda el: 1e-6 if el < 0 else el)
pred_test = pd.Series(model.predict(x_test_poly), index=y_test.index).apply(lambda el: 1e-6 if el < 0 else el)

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

