#!/usr/bin/env python
# coding: utf-8

# <img src="../utils/title.png" alt="title", width="600">

# <img src="../utils/master_203_pic.jpg" alt="Master 203", width=400>

# ## About Sberbank russian housing market challenge

# In 2017, Sberbank Russia’s oldest and largest bank has challenged data scientists from all around the world with a machine learning competiton ($25,000 prizes). In this competition, competitors (kagglers) had to develop algorithms which use a broad spectrum of features to predict russian housing market prices. Competitors rely on a rich dataset that includes housing data and macroeconomic patterns. Winning models have allowed Sberbank to provide more certainty to their customers in an uncertain economy. in this live-coding, we will experiment different agorithms and supervised machine learning pipeline features.
# 
# See more on : [Kaggle competition web page](https://www.kaggle.com/c/sberbank-russian-housing-market)

# # Lecture 1 - Linear and logistic regression
# 
# Let's run a simple linear regression to model russian housing market.

# ## Prerequisites

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Load Moscow housing data

# In[2]:


df = pd.read_csv("../data/train.csv")
print("dataframe shape : %s" % str(df.shape))


# ## Display first rows

# In[3]:


pd.set_option("display.max_columns", 500)
df.head(5)


# ## Available Features

# In[4]:


fil = open("../data/data_dictionnary.txt", "r")
print(fil.read())


# ## Train a simple linear regression between full_sq and price_doc

# In[5]:


from sklearn.linear_model import LinearRegression

# Train a simple linear regression algorithm
clf = LinearRegression()
model = clf.fit(df[["full_sq"]], df["price_doc"])
df["pred"] = model.predict(df[["full_sq"]])

# Displaty results
df[["full_sq", "price_doc", "pred"]].head(10)


# ## Access model coefficients

# In[6]:


model_bias = model.intercept_
model_coeff = model.coef_[0]
print("model intercept : %s" % model_bias)
print("model coeffs : %s" % model_coeff)


# ## Plot model

# In[7]:


plt.scatter(df["full_sq"], df["price_doc"])
plt.plot(np.linspace(0,6000, 1000), model.predict(np.linspace(0,6000, 1000).reshape(-1, 1)), color="g")
plt.xlabel("full_sq")
plt.ylabel("price doc (russian ruble)")
plt.show()


# ## Zoom on observations less than 400 square meters

# In[8]:


plt.scatter(df[df["full_sq"]<400]["full_sq"], df[df["full_sq"]<400]["price_doc"])
plt.plot(np.linspace(0,400, 1000), model.predict(np.linspace(0,400, 1000).reshape(-1, 1)), color="g")
plt.xlabel("full_sq")
plt.ylabel("price doc (russian ruble)")
plt.show()


# # Lecture 1 - Polynomial regression and Regularization
# 
# Let's try polynomial regression to model relationship between housing surface (full_sq) and price. Going further using regularization.

# ## Create full_sq polynomial features (polynomial order = 3)
# 
# Skelarn provides tools to create polynomial features.

# In[28]:


from sklearn.preprocessing import PolynomialFeatures

# Create polynomial featrues from variable full_sq :
polynomial_order = 3
poly = PolynomialFeatures(polynomial_order)
x_features = poly.fit_transform(df["full_sq"].reshape(-1,1))
x_features = pd.DataFrame(x_features, columns = ["full_sq^%s" % s for s in range(0, polynomial_order + 1)])
x_features.head(5)


# ## Standardize features
# 
# Once again, sklearn provides tools to standardize features in a really simple way. note that this a strong prerequisite to regularization.

# In[29]:


from sklearn.preprocessing import StandardScaler

# Standardize features using sklearn Standardscaler
clf = StandardScaler()
scaler = clf.fit(x_features)
x_features_scaled = pd.DataFrame(scaler.transform(x_features), columns = ["full_sq^%s" % s for s in range(0, polynomial_order + 1)])
x_features_scaled.head(5)


# ## Train a polynomial regression

# In[30]:


# Train a simple linear regression algorithm
clf = LinearRegression()
model = clf.fit(x_features_scaled, df["price_doc"])
df["pred"] = model.predict(x_features_scaled)

# Display results
df[["full_sq", "price_doc", "pred"]].head(10)


# ## Plots results (zoom on full_sq < 400)
# 
# Plot polynomial regression hypothesis function.

# In[31]:


# Create polynomial regression results (green curve)
x_axis = np.linspace(0,400, 400)
x_axis_poly = poly.fit_transform(x_axis.reshape(-1, 1))
x_axis_standardized = scaler.transform(x_axis_poly)

# plot
plt.scatter(df[df["full_sq"] < 400]["full_sq"], df[df["full_sq"] < 400]["price_doc"])
plt.plot(np.linspace(0, 400, 400), model.predict(x_axis_standardized), color="g")
plt.xlabel("full_sq")
plt.ylabel("price doc (russian ruble)")
plt.show()


# ## Take a look on residuals

# In[32]:


from sklearn.neighbors import KNeighborsRegressor

# Take a look on residuals and plot a KNN 
clf = KNeighborsRegressor(50)
model_knn = clf.fit(df[df["full_sq"] < 400][["full_sq"]], df[df["full_sq"] < 400]["price_doc"] - df[df["full_sq"] < 400]["pred"])
plt.scatter(df[df["full_sq"] < 400]["full_sq"], df[df["full_sq"] < 400]["price_doc"] - df[df["full_sq"] < 400]["pred"])
plt.plot(np.linspace(0, 500, 500), model_knn.predict(np.linspace(0, 500, 500).reshape(-1,1)), color="r")
plt.show()


# ## Same with polynomial order = 7
# 
# Let's try a higher polynomial order. As you can observe on below chart : 
# - Hypothesis function is now more complex than before.
# - Hypothesis function is completly overfitting for properties with more than 300 square meters (outliers?).

# In[33]:


# Create polynomial features
polynomial_order = 7
poly = PolynomialFeatures(polynomial_order)
x_features = poly.fit_transform(df["full_sq"].reshape(-1,1))
x_features = pd.DataFrame(x_features, columns = ["x^%s" % s for s in range(0, polynomial_order + 1)])
x_features.head(5)

# Standardize features using sklearn Standardscaler
scaler = StandardScaler()
scaler = scaler.fit(x_features)
x_features_scaled = pd.DataFrame(scaler.transform(x_features), columns = ["x^%s" % s for s in range(0, polynomial_order + 1)])

# Train a simple linear regression algorithm
clf = LinearRegression()
model = clf.fit(x_features_scaled, df["price_doc"])
df["pred"] = model.predict(x_features_scaled)

# Create polynomial regression results (green curve)
x_axis = np.linspace(0,400, 400)
x_axis_poly = poly.fit_transform(x_axis.reshape(-1, 1))
x_axis_standardized = scaler.transform(x_axis_poly)

# plot
plt.scatter(df[df["full_sq"] < 400]["full_sq"], df[df["full_sq"] < 400]["price_doc"])
plt.plot(np.linspace(0, 400, 400), model.predict(x_axis_standardized), color="g")
plt.xlabel("full_sq")
plt.ylabel("price doc (russian ruble)")
plt.show()


# ## Take a look on residuals

# In[34]:


from sklearn.neighbors import KNeighborsRegressor

# Take a look on residuals and plot a KNN 
clf = KNeighborsRegressor(50)
model_knn = clf.fit(df[df["full_sq"] < 400][["full_sq"]], df[df["full_sq"] < 400]["price_doc"] - df[df["full_sq"] < 400]["pred"])
plt.scatter(df[df["full_sq"] < 400]["full_sq"], df[df["full_sq"] < 400]["price_doc"] - df[df["full_sq"] < 400]["pred"])
plt.plot(np.linspace(0, 500, 500), model_knn.predict(np.linspace(0, 500, 500).reshape(-1,1)), color="r")
plt.show()


# ## Take a look on model coefficients

# In[35]:


model_bias = model.intercept_
model_coeff = pd.Series(model.coef_.tolist(), index=["full_sq^%s" % s for s in range(0, polynomial_order + 1)])
print("model intercept : %s" % model_bias)
print("model coeffs :\n%s" % model_coeff)


# ## Regularization
# 
# Let's try ridge regression (norm L2). We make lambda regularization strength coefficient and observe resulting hypothesis functions.

# In[36]:


from sklearn.linear_model import Ridge

lambda_range = [1e-10, 1e-7, 1e-5, 1e-3, 1.0, 100.0, 10000.0, 1000000]
for lamb in lambda_range:
    # Create polynomial features
    polynomial_order = 7
    poly = PolynomialFeatures(polynomial_order)
    x_features = poly.fit_transform(df["full_sq"].reshape(-1,1))
    x_features = pd.DataFrame(x_features, columns = ["x^%s" % s for s in range(0, polynomial_order + 1)])
    x_features.head(5)

    # Standardize features using sklearn Standardscaler
    scaler = StandardScaler()
    scaler = scaler.fit(x_features)
    x_features_scaled = pd.DataFrame(scaler.transform(x_features), columns = ["x^%s" % s for s in range(0, polynomial_order + 1)])

    # Train a simple Ridge regression algorithm
    clf = Ridge(lamb)
    model = clf.fit(x_features_scaled, df["price_doc"])
    df["pred"] = model.predict(x_features_scaled)

    # Create polynomial regression results (green curve)
    x_axis = np.linspace(0,400, 400)
    x_axis_poly = poly.fit_transform(x_axis.reshape(-1, 1))
    x_axis_standardized = scaler.transform(x_axis_poly)
    
    # coefficients evolution.
    model_bias = model.intercept_
    model_coeff = pd.Series(model.coef_.tolist(), index=["full_sq^%s" % s for s in range(0, polynomial_order + 1)])
    print("\n----- Ridge regression with lambda = %s" % lamb)
    print("model intercept : %s" % model_bias)
    print("model coeffs :\n%s" % model_coeff)
    # plot
    plt.scatter(df[df["full_sq"] < 400]["full_sq"], df[df["full_sq"] < 400]["price_doc"])
    plt.plot(np.linspace(0, 400, 400), model.predict(x_axis_standardized), color="g")
    plt.xlabel("full_sq")
    plt.ylabel("price doc (russian ruble)")
    plt.title("Ridge regression : lambda = %s" % lamb)
    plt.show()

