#!/usr/bin/env python
# coding: utf-8

#  ## Note: my PC died during the competition so I lost a lot of code and I was unable to reconstruct a full notebook representing the full extent of my work and sweat.
#  
#  As a consequence, some models are not included like random forest as well as some graphical analysis.
#  
#  
#  

# # Prerequisites

# ## Aim of the competition

# Among capital markets' activity spectrum, the Equity Derivatives department creates investment, hedging and multipurpose solutions for professional clients. These solutions use exotic options based on one or more baskets of underlyings among which stocks, indices and other risky vehicles. Those exotic options need complex numerical algorithms to be priced, using Monte-Carlo methods in a non-trivial way due to the path-dependent nature of the pay-off and the callability of the instrument. The typical time of a single pricing can go between a few seconds and a few minutes on a farm of servers.<br/>
# 
# The purpose of the challenge is to use supervised machine learning to learn how to price a specific type of instruments described by the following dataset. The benefit would be to singularly accelerate computation time while retaining good pricing precision. The exotic option to price is a path-dependent option whose final payoff is conditional on the path of a basket of three stocks or equity indices.

# ## About data

# Data consists of 32k exotic options pricings for the train dataset and 8k observations for the test set. Both datasets contain the same columns (except column **target** you have to predict for test dataset. As a competitor, you have to train your best machine learning algorithm to model the price of such exotic options.
# 
# For each option are provided 15 variables : 
# 
# <ul>
# <li><strong>id</strong> - unique sample identifier</li>
# <li><strong>s1</strong> - Underlying 1</li>
# <li><strong>s2</strong> - Underlying 2</li>
# <li><strong>s3</strong> - Underlying 3</li>
# <li><strong>mu1</strong> - Average drift of the underlying 1</li>
# <li><strong>mu2</strong> - Average drift of the underlying 2</li>
# <li><strong>mu3</strong> - Average drift of the underlying 3</li>
# <li><strong>v1</strong> - Volatility of the underlying 1</li>
# <li><strong>v2</strong> - Volatility of the underlying 2</li>
# <li><strong>v3</strong> - Volatility of the underlying 3</li>
# <li><strong>c12</strong> - Correlation underlying 1 - underlying 2</li>
# <li><strong>c13</strong> -  Correlation underlying 1 - underlying 3</li>
# <li><strong>c23</strong> -  Correlation underlying 2 - underlying 3</li>
# <li><strong>strike</strong> - Strike.</li>
# <li><strong>t</strong> - Maturity of the deal.</li>
# </ul>
# 

# ## Load modules
# 
# In this cell, you can put all modules you use. Yoy can use it to provide a clear code.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt


# In[3]:


#NN 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model

#MSE
from sklearn.metrics import mean_squared_error

#Tree regression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

#Linear regression
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import AlphaSelection

#For Gradient boosting
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier

#For features
from yellowbrick.features.importances import FeatureImportances
from yellowbrick.features.rankd import Rank1D, Rank2D
from yellowbrick.features.radviz import RadViz
from yellowbrick.features.pcoords import ParallelCoordinates
from yellowbrick.features.jointplot import JointPlotVisualizer
from yellowbrick.features.pca import PCADecomposition
from yellowbrick.features.manifold import Manifold
from yellowbrick.features.importances import FeatureImportances
from yellowbrick.features.rfecv import RFECV

#For target
from yellowbrick.target import BalancedBinningReference
from yellowbrick.target import ClassBalance
from yellowbrick.target import FeatureCorrelation


# ## Your functions
# 
# In order to have a clear code, you can put all your own functions in this cell.

# In[4]:


def export_kaggle(df_test, pred_test, save=True, path_save="my_submission_vanilla_model.csv"):
    """
    Export submissions with the good kaggle format.
    df_test : (pandas dataframe) test set
    proba_test : (numpy ndarray) probabilities as numpy ndarray you get using method .predict()
    save : (bool) if set to True, it will save csv submission in path_save path.
    path_save : (str) path where to save submission.
    return : dataframe for submission
    """
    pred_serie = pd.Series(pred_test, index=df_test.index)
    df_submit = pd.concat([df_test["id"], pred_serie], axis=1)
    df_submit.columns = ["id", "target"]
    df_submit.to_csv(path_save, index=False)
    return df_submit

def check_test(result, expected, display_error):
    """
    Testing your results.
    """
    if result == expected:
        print("1 test passed.")
    else:
        print(display_error)


# #  Data Exploration and Features engineering/selection
# 
# Data exploration is a common part in a machine learning pipeline. In this section, you will import datasets, discover  features, provide data mining observations, investigate missing values and possible outliers. An exhaustive exploration is more likely to yield prowerful predictive models.
# 
# Here we will combine both sections as we feel they are binded.

# ## Load datasets

# In[5]:


df_train = pd.read_csv("C:/Users/youssef/OneDrive/Documents/MASTER203/Machine Learning/lab2/data/df_train.csv")
labels_train = pd.read_csv("C:/Users/youssef/OneDrive/Documents/MASTER203/Machine Learning/lab2/data/y_train.csv")
df_test = pd.read_csv("C:/Users/youssef/OneDrive/Documents/MASTER203/Machine Learning/lab2/data/df_test.csv")
n_rows_train = df_train.shape[0]
n_rows_test = df_test.shape[0]


# In[6]:


check_test(n_rows_train, 32000, "wrong number of rows")
check_test(n_rows_test, 8000, "wrong number of rows")


# ## Take a look on first rows
# 
# Take a look on the **df_train** first 5 rows using method .head().

# In[6]:


print(df_train.head())


# Take a look on the **labels_train** first 5 rows using method .head().

# In[7]:


print(labels_train.head())


# ## Look at the distribution of the target
# 
# Let's take a look on the column label (target) in the train dataset. Take a look on the disribution of the label by plotting an histogram with matplotlib.
# 
# - Plot the histogram of the serie **labels_train**. 
# - Describe the serie **labels_train** using the method **.describe()**.

# In[8]:


# Plot the histogram
plt.hist(labels_train["target"],bins=100)
plt.show()

# Describe the serie of label with the method .describe()
labels_train["target"].describe()


# ## Correlation between the strike and the price of the option
# 
# (Here is an example of data exploration + interpretation you can perform to understand your features).
# 
# The challenge only contains call options. Due to the structure of the options to price, we can assert that there is a  negative correlation between the strike of the option and its price. 
# 
# - Using the module matplotlib, make a scatter plot where the x-axis is the strike of the option and the y-axis is the price (target) of the option.
# - Using the method .corr(), compute the correlation between the stike and the price of the option. Assign the result to the variable **corr_strike_price**.

# In[9]:


# Plot the pdistrike on the x axis and the price of the option on the y-axis.
plt.scatter(df_train["strike"], labels_train["target"])
plt.xlabel("strike")
plt.ylabel("price")
plt.show()

# Compute the correlation between the two variables.
corr_strike_price = df_train["strike"].corr(labels_train["target"])
print(corr_strike_price)


# In[10]:


check_test(round(abs(corr_strike_price), 4), 0.6072, "wrong correlation between pdistrike and target.")


# ## Distribution of the spot prices
# 
# The option payoff is based on a worst strategy. It means that the option payoff is related to the performance of the worst of the three underlyings. Let's try to take a look on the distribution of the spread between the underlying with the larger spot price and the underlying with the smallest spot price.
# 
# - For each obseration, get the largest spot price between the three underlyings and assign the resulting serie to the variable **serie_spot_max**.
# - For each obseration, get the smallest spot price between the three underlyings and assign the resulting serie to the variable **serie_spot_min**.
# - Compute the difference between **serie_spot_max** and **serie_spot_min** and assign the result to the variable **serie_spot_spread**.
# - and plot the histogram of the serie **serie_spot_spread**.

# In[11]:


serie_spot_max = df_train.loc[:,["s1","s2","s3"]].apply(lambda x: x.max(), axis=1)
serie_spot_min = df_train.loc[:,["s1","s2","s3"]].apply(lambda x: x.min(), axis=1)
serie_spot_spread = serie_spot_max - serie_spot_min

# Plot
plt.hist(serie_spot_spread,bins=100)
plt.show()


# In[12]:


#Obtain relevant descriptive stats
data = df_train.iloc[:,1:]
data["target"] = labels_train["target"]

data.describe()


# # Correlation check between variables

# In[13]:


# Let's look at the correlation between our variables
# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation plot')


# One can confirm the previous observation made on the negative correlation between the strike and the price (target) thanks to the above correlation plot. We also can note that the variables are not significantly correlation between one another which is good at it avoids some bias in our regressions to be made. Although many of them exhibit some correlation with the variable to be predicted, i.e. the price of the option, which means it will add some information to obtain a good prediction. It is noteworthy to say that the dataset seems to be standardized with no missing values and therefore reduces the need for intervention in data preliminary treatment.

# ## Statistical Inference
# 
# Here will be conducted a linear regression in order to check the statistical inference of our dependant variables that will be used to predict the option price.

# In[14]:


X = sm.add_constant(data.iloc[:,:-1])
model = sm.OLS(data.iloc[:,-1],data.iloc[:,:-1])
results = model.fit()
results.summary()


# We can see from the above regression summary that all the variables are considered statistically significant and could therefore be relevant to our predictive model. We will therefore keep for now all variables for the rest of the analysis as they seem to bring information to predict the target.

# ## Multiply the volatility with the maturity
# 
# In order to help the learning, we can create a more advanced feature by multiplying the volatilities with the square root of the maturity.
# - Create the features **vol_sq_t_1**, **vol_sq_t_2** and **vol_sq_t_3** by multiplying the volatility of each underlying with the square root of the maturity.
# - apply the process for both train and test set.

# In[7]:


# Train set
df_train["vol_sq_t_1"] = df_train["v1"]*df_train["t"].apply(lambda x: sqrt(x))
df_train["vol_sq_t_2"] = df_train["v2"]*df_train["t"].apply(lambda x: sqrt(x))
df_train["vol_sq_t_3"] = df_train["v3"]*df_train["t"].apply(lambda x: sqrt(x))

# Test set
df_test["vol_sq_t_1"] = df_test["v1"]*df_test["t"].apply(lambda x: sqrt(x))
df_test["vol_sq_t_2"] = df_test["v2"]*df_test["t"].apply(lambda x: sqrt(x))
df_test["vol_sq_t_3"] = df_test["v3"]*df_test["t"].apply(lambda x: sqrt(x))

print(df_train.head())


# #  Features analysis
# 
# Correlation plot above gave us already some information, but we can used more advanced tools at our disposal to sort the features. Below are some interesting methods that will allow us to familirize ourselves with the existing features.

# In[16]:


# Harmonize a dataset to use
data = df_train.iloc[:,1:]
data["target"] = labels_train["target"]

X = data.iloc[:,:-1]
y = data["target"]

features = X.columns


# In[17]:


# Instantiate the 1D visualizer with the Sharpiro ranking algorithm
visualizer = Rank1D(features=features, algorithm='shapiro')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[18]:


# Instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=features, algorithm='covariance')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# From the above Cov ranking, we can see it is almost null for each pair of variables, so it strenghtens our initial idea to keep al of them.

# In[19]:


# Instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=features, algorithm='pearson')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# Same conjecture to be done here. Only strong correlation between vol_sq_t_i and vi and t variables which makes sense as the last 3 variables have been created as a combinaison of the latter 2 variables mentionned.

# In[20]:


# Instantiate the visualizer
visualizer = BalancedBinningReference()

visualizer.fit(y)          # Fit the data to the visualizer
visualizer.poof()          # Draw/show/poof the data


# In[8]:


# Select the final features to be used 
# i.e. all of them...
features = df_train.columns[1:]


# In[9]:


check_test("id" not in features, True, "error : column id still in the features.")


# # Machine learning algorithms experiments

# ## Validation procedure
# 
# In this section, we will split our dataset into a train set and a validation set using the train_test_split approach. 

# In[10]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(df_train[features],labels_train["target"],test_size=0.2,random_state=203)


# Here we split our dataset as it is important to test the predictive power of our model in a validation set.

# ## RIdge regression
# 
# Let us test a first ridge regression model with default parameters and all the features using the sklearn package.
# Indeed, we already used a linear regression above, and from the what we obtained above in a first overview, a ridge regression seems more appropriate given the number of variables we have. If estimates (β) values are very large, then the SSE term will minimize, but the penalty term will increases. If estimates(β) values are small, then the penalty term will minimize, but, the SSE term will increase due to poor generalization. So, it chooses the feature's estimates (β) to penalize in such a way that less influential features (Some features cause very small influence on dependent variable) undergo more penalization. In some domains, the number of independent variables is many, as well as we are not sure which of the independent variables influences dependent variable. In this kind of scenario, ridge regression plays a better role than linear regression. Although, the prediction power is not expected to be that different of the linear regression.
# 

# In[24]:


#Linear Regression
# Fit a simple linear regression model
clf = Ridge()
model = clf.fit(x_train, y_train)

# Make prediction on train, validation and test set.
pred_train = model.predict(x_train)
pred_val = model.predict(x_val)

# Evaluate the results of the regression
mse_train = mean_squared_error(y_train, pred_train)
mse_val = mean_squared_error(y_val, pred_val)

print("MSE score on train dataset : %s" % mse_train)
print("MSE score on validation dataset : %s" % mse_val)


# This is the first score we obtain for our prediction using the Ridge regression. Obviously it can be improved using more powerful models but already gives us a benchmark to beat from here on.

# In[25]:


# Instantiate the linear model and visualizer
visualizer = ResidualsPlot(clf)

visualizer.fit(x_train, y_train)  # Fit the training data to the model
visualizer.score(x_val, y_val)  # Evaluate the model on the validation data
visualizer.poof()  


# # Tree regression
# 
# In order to do thing properly for this model, we will try and pick the paramters that fit best.
# - First the max depth
# - Second the min samples split
# - Third the min samples leaf
# 

# In[26]:


"""
MAX DEPTH
"""
depths = range(1, 25)
all_mse_train = []
all_mse_val = []
experiments  = []

for depth in depths: 
    clf = DecisionTreeRegressor(max_depth=depth)
    model = clf.fit(x_train, y_train)

    # Make prediction on train, validation and test set.
    pred_train = pd.Series(model.predict(x_train), index=y_train.index)
    pred_val = pd.Series(model.predict(x_val), index=y_val.index)

    # Compute MSE evaluation metrics
    mse_train = mean_squared_error(y_train, pred_train)
    mse_val = mean_squared_error(y_val, pred_val)
    all_mse_train.append(mse_train)
    all_mse_val.append(mse_val)
    print("depth = %s | MSE train = %s | MSE val = %s " % (depth, mse_train, mse_val))

plt.plot(depths, all_mse_train, color="b")
plt.plot(depths, all_mse_val, color="g")
plt.xlabel("tree depth")
plt.ylabel("MSE")
plt.title("Learning curves - playing on tree depth")
plt.show()


# Looking at the plot, we decide to pick a depth = 15 as the MSE seems to have almost fully converged towards its min at that level.

# In[27]:


"""
MIN SAMPLES SPLIT (1/2)
"""

depths = range(1, 25)
min_samples = [2, 6, 10, 15, 20, 30, 50, 70]
all_mse_train = []
all_mse_val = []

for min_s in min_samples:
    temp_mse_train = []
    temp_mse_val = []
    min_mse_val = 10
    best_conf = 0
    print("Compute trees with min_samples_splits = %s" % min_s)
    for depth in depths: 
        clf = DecisionTreeRegressor(max_depth=depth, min_samples_split=min_s)
        model = clf.fit(x_train, y_train)

        # Make prediction on train, validation and test set.
        pred_train = pd.Series(model.predict(x_train), index=y_train.index)
        pred_val = pd.Series(model.predict(x_val), index=y_val.index)

        # Compute MSE evaluation metrics
        mse_train = mean_squared_error(y_train, pred_train)
        mse_val = mean_squared_error(y_val, pred_val)
        temp_mse_train.append(mse_val)
        temp_mse_val.append(mse_val)
        
        # Checking minimum
        if mse_val < min_mse_val:
            min_mse_val = mse_val
            best_conf = depth
    print("--> Min MSE on validation set = %s for depth = %s" % (min_mse_val, best_conf))
    all_mse_train.append(temp_mse_train)
    all_mse_val.append(temp_mse_val)


# In[28]:


"""
MIN SAMPLES SPLIT (2/2)
"""
# Make plots
for (curve, min_sample) in zip(all_mse_val, min_samples):
    plt.plot(depths, curve, label="min samples = %s" % min_sample)
plt.xlabel("tree depth")
plt.ylabel("MSE")
plt.legend(loc="upper left", prop={'size': 8})
plt.title("Min samples splits impact on validation set")
plt.show()

for (curve, min_sample) in zip(all_mse_val, min_samples):
    plt.plot(depths[12:18], curve[12:18], label="min samples = %s" % min_sample)
plt.xlabel("tree depth")
plt.ylabel("MSE")
plt.legend(loc="upper left", prop={'size': 8})
plt.title("Min samples splits impact on validation set - zoom")
plt.show()


# Here again, we get an indication of the min samples to use and clearly from the above graph, 2 is preferred.

# In[29]:


"""
MIN SAMPLES LEAF (1/2)
"""
# Fit a simple linear regression model
depths = range(1, 25)
min_leaf = [1, 10, 30, 50, 100, 300, 1000]
all_mse_train = []
all_mse_val = []

for min_l in min_leaf:
    temp_mse_train = []
    temp_mse_val = []
    min_mse_val = 10
    best_conf = 0
    print("Compute trees with min_samples_leaf = %s" % min_l)
    for depth in depths: 
        clf = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_l)
        model = clf.fit(x_train, y_train)

        # Make prediction on train, validation and test set.
        pred_train = pd.Series(model.predict(x_train), index=y_train.index)
        pred_val = pd.Series(model.predict(x_val), index=y_val.index)

        # Compute MSE evaluation metrics
        mse_train = mean_squared_error(y_train, pred_train)
        mse_val = mean_squared_error(y_val, pred_val)
        temp_mse_train.append(mse_val)
        temp_mse_val.append(mse_val)
        
        # Checking minimum
        if mse_val < min_mse_val:
            min_mse_val = mse_val
            best_conf = depth
    print("--> Min MSE on validation set = %s for depth = %s" % (min_mse_val, best_conf))
    all_mse_train.append(temp_mse_train)
    all_mse_val.append(temp_mse_val)


# In[30]:


"""
MIN SAMPLES LEAF (2/2)
"""

# Make plots
for (curve, min_l) in zip(all_mse_val, min_leaf):
    plt.plot(depths, curve, label="min samples per leaf = %s" % min_l)
plt.xlabel("tree depth")
plt.ylabel("MSE")
plt.legend(loc="upper left", prop={'size': 8})
plt.title("Min samples splits impact on validation set")
plt.show()

for (curve, min_l) in zip(all_mse_val, min_leaf):
    plt.plot(depths[12:18], curve[12:18], label="min samples per leaf = %s" % min_l)
plt.xlabel("tree depth")
plt.ylabel("mse")
plt.legend(loc="upper left", prop={'size': 8})
plt.title("Min samples splits impact on validation set - zoom")
plt.show()


# Same as before, here the mse is minimized for a tree depth of 15 with a min samples leaf equal to 1.

# In[31]:


# Tree regression
clf = DecisionTreeRegressor(max_depth=15, min_samples_split=2, min_samples_leaf=1)
model = clf.fit(x_train,y_train)
pred_train = model.predict(x_train)
pred_val = model.predict(x_val)

# Evaluate the results of the regression
mse_train = mean_squared_error(y_train, pred_train)
mse_val = mean_squared_error(y_val, pred_val)

print("MSE score on train dataset : %s" % mse_train)
print("MSE score on validation dataset : %s" % mse_val)


# Clearly the tree regression did better than the Ridge regression with a huge improvement on the MSE score.

# In[32]:


# Instantiate the tree model and visualizer
visualizer = ResidualsPlot(clf)

visualizer.fit(x_train, y_train)  # Fit the training data to the model
visualizer.score(x_val, y_val)  # Evaluate the model on the validation data
visualizer.poof() 


# # Gradient boosting

# In[33]:


params = {'criterion':"friedman_mse", 'init':None,
'learning_rate':0.12,'max_depth':5,
'max_features':None, 'max_leaf_nodes':None, 'min_samples_leaf':1,
'min_samples_split':2, 'min_weight_fraction_leaf':0.0,
'n_estimators':220, 'presort':"auto", 'random_state':None,
'subsample':1.0, 'verbose':0, 'warm_start':False}

gradient_boosting_regressor = ensemble.GradientBoostingRegressor(**params)
gradient_boosting_regressor.fit(x_train, y_train)

pred_train = gradient_boosting_regressor.predict(x_train)
pred_val = gradient_boosting_regressor.predict(x_val)

# Evaluate the results of the regression
mse_train = mean_squared_error(y_train, pred_train)
mse_val = mean_squared_error(y_val, pred_val)

print("MSE score on train dataset : %s" % mse_train)
print("MSE score on validation dataset : %s" % mse_val)


# Here again, great improvement of the MSE score compared to the previous models. It is our favorite for now.

# In[34]:


# Instantiate the tree model and visualizer
visualizer = ResidualsPlot(clf)

visualizer.fit(x_train, y_train)  # Fit the training data to the model
visualizer.score(x_val, y_val)  # Evaluate the model on the validation data
visualizer.poof() 


# ## Neural network regression
# 
# Here we will try to find the best neural network. Unfortunatel, as previousl said, my computer died and I therefore lost a lot of pieces of code that were not saved on my cloud for capacity constraints (awesome timing). So I will only submit one the version I have that does not include many of the things I tried or useful tools (like callbacks).
# Many sets of features have been tested also, and combination of hyperparamters, to determine the best set of neurones, batch size ...etc but again I am not able to showcase all the work that was embbeded in this competition.
# In the hope that it will be taken into consideration.
# 
# **Note : I am not able to re-run the models in the fear of ruining my friend's computer as well :/ !**
# 
# The following models are therefore set with epochs = 10 instead of 1000 that I actually used.

# In[11]:


# Set the path to save the models
path = "C:/Users/youssef/OneDrive/Documents/MASTER203/Machine Learning/lab2/models/"


# In[12]:


# Let us first define a function to run the Sequential model from Keras
def NeuralNetwork(hidden_layers, activations, epochs, batch_size, optimizer, verbose, show_results):
    model = Sequential()
    # First input layer
    model.add(Dense(hidden_layers[0], activation = activations[0], input_dim = len(features)))
    # Loop over the desired number of layers
    for i in range(1, len(hidden_layers)):
        model.add(Dense(hidden_layers[i], activation = activations[i]))
    # Last output layer
    model.add(Dense(1, activation = 'linear'))
    # Compilation of the model
    model.compile(loss = 'mean_squared_error', optimizer = optimizer)
    # Fit to train the model
    result = model.fit(x_train, y_train, #training data to be used to train the model
              validation_data=(x_val, y_val), #to show the val_loss at the same time as the loss
              epochs = epochs, #number of epochs to train the data
              batch_size = batch_size, #number of batch to be used
              verbose = verbose) #to show the ongoing loss measures as the training occurs
    # Compute prediction output    
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)
    # Compute MSE
    mse_train = mean_squared_error(pred_train, y_train)
    mse_val = mean_squared_error(pred_val, y_val)
    # Show the results if wanted
    if show_results == True:
        print("MSE score on train dataset : %s" % mse_train)
        print("MSE score on validation dataset : %s" % mse_val)
        # Plot to see the convergence of the val loss
        plt.plot(result.history['val_loss'])
        print("Min score : %f at %0.f epochs" % (min(result.history['val_loss']),
                                                 list(result.history['val_loss']).index(min(result.history['val_loss']))))
    
    else:
        print("You decided not to show the results, if you meant to show them please input show_results = True.")
        
    return model   


# In[13]:


# Model 1
#inputs
verbose = 1
batch_size = 64
epochs = 10
show_results = True
hidden_layers = [700, 500, 400, 300, 200, 100, 50]
activations = np.repeat('relu', len(hidden_layers))

# Train the model
model1 = NeuralNetwork(hidden_layers, activations, epochs, batch_size, 'adamax', 1, 1)
model1.save(path +'model1.h5')


# In[14]:


# Model 2
#inputs
verbose = 1
batch_size = 64
epochs = 10
show_results = True
hidden_layers = [600, 400, 200, 100, 75, 50, 20]
activations = np.repeat('relu', len(hidden_layers))

# Train the model
model2 = NeuralNetwork(hidden_layers, activations, epochs, batch_size, 'adamax', 1, 1)
model2.save(path +'model_2.h5')


# In[15]:


# Model 3
#inputs
verbose = 1
batch_size = 64
epochs = 10
show_results = True
hidden_layers = [700, 350, 175, 90, 45, 20]
activations = np.repeat('relu', len(hidden_layers))

# Train the model
model3 = NeuralNetwork(hidden_layers, activations, epochs, batch_size, 'adamax', 1, 1)
model3.save(path +'model_3.h5')


# In[16]:


#If the models need to be trained some more, here is to load them
#model1 = load_model(path + 'model_1.h5')
#model2 = load_model(path + 'model_2.h5')
#model3 = load_model(path + 'model_3.h5')


# # Final run
# 
# According to previous steps results and your own interpretation, run your best machine learning algorithm and make  predictions on the test set. Then export your results in and make submission on Kaggle platform.

# ## Run your best model and make prediction on test set
# 
# For this first experiment, use the vanilla linear regresson we have experimented in step 3. Make predictions on the test set and assign the results to variable **pred_test**.

# In[17]:


# Predict with the best model
pred_train_M1 = model1.predict(x_train)
pred_val_M1 = model1.predict(x_val)
pred_test_M1 = model1.predict(df_test.iloc[:,1:])

pred_train_M2 = model2.predict(x_train)
pred_val_M2 = model2.predict(x_val)
pred_test_M2 = model2.predict(df_test.iloc[:,1:])

pred_train_M3 = model3.predict(x_train)
pred_val_M3 = model3.predict(x_val)
pred_test_M3 = model3.predict(df_test.iloc[:,1:])

# Best model, average of our 3 models
pred_train = (pred_train_M1 + pred_train_M2 + pred_train_M3) / 3
pred_val = (pred_val_M1 + pred_val_M2 + pred_val_M3) / 3

pred_test = (pred_test_M1 + pred_test_M2 + pred_test_M3) / 3
pred_test = pd.Series(pred_test[:,0])


# In[24]:


# Plot the residuals
res = pred_val[:,0] - y_val
plt.scatter(pred_val[:,0], res)
plt.axhline(linewidth=1, color='r')


# In[25]:


check_test(pred_test.shape, (8000,), "wrong shape for pred test")
len(pred_test)


# ## Evaluate best model
# 
# Our best machine learning algorithm yields the following results : 

# In[43]:


mse_train = mean_squared_error(y_train, pred_train)
mse_val = mean_squared_error(y_val, pred_val)
print("mse score on train dataset : %s" % mse_train)
print("mse score on validation dataset : %s" % mse_val)


# ## Interpretation

# Throughout the competition, we soon learned that neural network is the best suited model, and my results (although not shown on this notebook for the above-mentionned reasons) clearly demonstrate the efficiency and predictive power of the neural network. The selection of hyperparameters was at first sight made following recommendations on what could have been read on the internet. And then in a more arbitrary way depending on the results obtained, always choosing the MSE as a decision criterion. Using the plot of val loss through the training, the selection of the number of epochs was done and epochs = 1000 and 1200 was optimal. Batch size as well was chosen with a learning curve (also unfortunately not in the notebook) and 32 to 64 were the best. I have choosen 64 for speed of convergence reasons. Decreasing the number of neurons with respect to the layers was also the best configuration. 
# 
# To sum up, neural network is the best way to predict such option prices but takes time to find the best configuration. Also, as seen above, I have decided to average 3 best models as it naturally helps reducing the MSE. Ideally, I wanted to build confidence interval for each models to make sure not to icnrease the variance too much when averaging and avoid as well too much covariantion between the models.

# ## Export submission to Kaggle
# 
# You can export your predictions as a submission file using function export_kaggle() defined earlier in the notebook. Then go on Kaggle competition page and submit your predictions. Your can observe your performance on leaderboard page.

# In[44]:


df_submit = export_kaggle(df_test, pred_test, True, "C:/Users/youssef/OneDrive/Documents/MASTER203/Machine Learning/lab2/results/final_submission.csv")


# ## Conclusion

# You can write your work conclusion in this cell..

# ## Other things you can try
# 
# If you lack in ideas, here are some you can try :
# 
# - Evaluate final model on train, validation and test set. Diagnose your model results.
# - By playing on decision boundary threshold, assign label to probabilities predictions.
# - Using thresholds, display confusion matrix for different thresholds.
# - take a look on other relevant evaluation metrics with regards to usecase. Justify.
# - Take a look on features importances.
# - Give conclusion;
# - ...
# 
# Feel free to provide your own interpretations and justifications. Here is your notebook and work !

# In[ ]:




