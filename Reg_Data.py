#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as bb

db=bb.read_csv(r'C:\ya1\a1\kc_house_data.csv')
db.drop('id',axis=1,inplace=True)


# In[2]:


db


# In[3]:


db['zipcode'].value_counts()
db['price'].value_counts()
db['bedrooms'].value_counts()


# In[4]:


db['bathrooms'].value_counts()


# In[5]:


db.corr()


# In[6]:


db.info()


# In[7]:


db.corr()


# In[3]:


db.drop('sqft_lot',axis=1,inplace=True)


# In[4]:


db.drop('condition',axis=1,inplace=True)


# In[5]:


db


# In[11]:


sns.heatmap(db.corr())


# In[12]:


sns.heatmap(db.corr(),cmap='coolwarm',annot=True)


# In[13]:


db.describe()


# In[6]:


db.groupby(['bedrooms']).count()


# In[7]:


sns.barplot(x='price',y='bedrooms',data=db)
#at 9 bedrooms the price is at the most


# In[8]:


sns.barplot(x='bathrooms',y='price',data=db)


# In[9]:


sns.barplot(x='sqft_living',y='price',data=db)


# In[10]:


sns.countplot(x='zipcode',data=db)
#zip code is errelative


# In[12]:


sns.distplot(db['price'],kde=False,bins=30)


# In[14]:


sns.jointplot(x='sqft_living',y='price',data=db,kind='scatter')


# In[15]:


sns.pairplot(db)


# In[16]:


sns.pairplot(db,hue='price',palette='coolwarm')


# In[17]:


db.pivot_table(values='price',index='sqft_living',columns=['grade'])


# In[18]:


piv = db.pivot_table(values='sqft_living',index='price',columns=['grade'])
sns.heatmap(piv)


# In[27]:


import pandas as New_DB
New_DB=db[['price','sqft_living','grade','bedrooms']]
New_DB.head()


# In[28]:


sns.pairplot(New_DB)


# In[32]:


X = db[['sqft_living','grade','bedrooms']]
y = db['price']
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[33]:


X_train


# In[34]:


X_test


# In[35]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[36]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)


# In[38]:


#Validation
import numpy as np

from sklearn import metrics
print("MSE:",metrics.mean_squared_error(y_pred,y_test))
print("MAE:",metrics.mean_absolute_error(y_pred,y_test))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_pred,y_test)))
print("r2_score:",metrics.r2_score(y_pred,y_test))


# In[ ]:




