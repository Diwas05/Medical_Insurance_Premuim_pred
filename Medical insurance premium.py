#!/usr/bin/env python
# coding: utf-8

# # MEDICAL INSURANCE PREMIUM PREDICTION MODEL

# In[1]:


# Here we are going to predict the premium of the insurance. Therefore "CHARGES" will be the target/dependent variable and rest all will be the independent variable.


# # PART 1: DATA PREPROCESSING

# # #1.1 Importing the libraries and Dataset

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv('insurance.csv')


# # #1.2 Data Exploration

# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


# Columns with categorical data
data.select_dtypes(include = 'object').columns
# After observation there are 3 categorical columns.


# In[8]:


# Length of categorical columns.
len(data.select_dtypes(include = 'object').columns)


# In[9]:


# Columns with numerical columns.
data.select_dtypes(include = ['int64','float64']).columns


# In[10]:


# length of numerical columns.
data.select_dtypes(include = ['int64','float64']).columns


# In[11]:


# STATISTICAL SUMMARY
data.describe()


# Group the dataset by 'sex','smoker','region' (CATEGORICAL DATA)

# In[12]:


data.groupby('sex').mean()


# In[13]:


data.groupby('smoker').mean()


# In[14]:


data.groupby('region').mean()


# # #1.3 Dealing with missing values

# In[15]:


data.isnull().values.any()
# False means that the dataset doesn't have any null values


# In[16]:


data.isnull().values.sum()


# # #1.4 Encoding the categorical data

# In[17]:


data.select_dtypes(include='object').columns


# In[18]:


# Finding the unique values
data['sex'].unique()


# In[19]:


data['smoker'].unique()


# In[20]:


data['region'].unique()


# In[21]:


data.head()


# In[22]:


# ONE HOT ENCODING
data=pd.get_dummies(data=data, drop_first=True)
# drop_first will drop the first column after one hot encoding to ovecome overfitting.


# In[23]:


data.head()


# In[24]:


data.shape


# # #1.5 Correlation Matrix

# In[25]:


# Now we are dropping the target variable (charges) and storing in the new variable to find the coorelation.
dataset = data.drop(columns='charges')


# In[26]:


dataset.head()


# In[27]:


# Now checking the coorealtion with target varibale(charges) and other independent variables.
dataset.corrwith(data['charges']).plot.bar(
    figsize=(16,9), title='Correlation with Charges', rot=45, grid=True
)


# AS WE CAN SEE THAT SMOKER IS WIDELY RELATED WITH CHARGES.

# In[28]:


corr = data.corr()


# In[29]:


# HEAT MAP
plt.figure(figsize=(16,9))
sns.heatmap(corr, annot=True)


# # #1.6 Splitting the dataset

# In[30]:


# matrix of features / independent variables
x =data.drop(columns='charges')


# In[31]:


# target / dependent variables
y = data['charges']


# In[32]:


# Now to split the data goto sci-kit docs./API then inside MODEL SELECTION 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[33]:


x_train.shape


# In[34]:


y_train.shape


# In[35]:


x_test.shape


# In[36]:


y_test.shape


# # #1.7 Feature Scaling
# we apply feature scaling bcz we want all the independent variable on the same scale

# In[37]:


# Standardize features by removing the mean and scaling to unit variance.

from sklearn.preprocessing import StandardScaler

# The standard score of a sample x is calculated as:

# z = (x - u) / s


# In[38]:


# Creating the instance of class
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[39]:


x_train


# In[40]:


x_test


# # PART 2: BUILDING THE MODEL

# # # 1) Multiple Linear Regression

# In[41]:


# class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
from sklearn.linear_model import LinearRegression


# In[42]:


#Creating the instance of class 
regressor_lr = LinearRegression()
regressor_lr.fit(x_train,y_train)


# In[43]:


# Prediction
y_pred = regressor_lr.predict(x_test)


# In[44]:


# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
# In the general case when the true y is non-constant, a constant model that always predicts the average y disregarding the input features would get a r2 score of 0.0.
from sklearn.metrics import r2_score


# In[45]:


# coefficient of determination
r2_score(y_test,y_pred)


# # # 2) Random Forest Regressor

# In[46]:


from sklearn.ensemble import RandomForestRegressor


# In[47]:


regressor_rf = RandomForestRegressor()
regressor_rf.fit(x_train,y_train)


# In[48]:


y_pred = regressor_rf.predict(x_test)


# In[49]:


# coefficient of determination
r2_score(y_test,y_pred)


# Therefore Random Forest Regressor is working better.

# # # 3) XGBoost Regression

# In[52]:


from xgboost import XGBRFRegressor


# In[51]:


pip install xgboost


# In[53]:


regressor_xgb = XGBRFRegressor()
regressor_xgb.fit(x_train,y_train)


# In[54]:


y_pred = regressor_xgb.predict(x_test)


# In[55]:


# coefficient of determination
r2_score(y_test,y_pred)


# Therefore Xgboost is working better

# # PART 3: PREDICT CHARGES FOR A NEW CUSTOMER

# # # Example 1:

# Name:Frank, age:40, sex:1,bmi:45.50, children:4, smoker:1, region:northeast

# In[56]:


data.head()


# In[57]:


frank_obs = [[40,45.5,4,1,1,0,0,0]]


# In[58]:


# Prediction
regressor_xgb.predict(sc.transform(frank_obs))


# 
