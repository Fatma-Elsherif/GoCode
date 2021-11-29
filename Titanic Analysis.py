#!/usr/bin/env python
# coding: utf-8

# In[10]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as bb

db=bb.read_csv(r'C:\ya1\a1\titanic-passengers.csv')


# In[12]:


db


# In[13]:


db=bb.read_csv(r'C:\ya1\a1\titanic-passengers.csv')
db=db[['Sex','Age','SibSp','Parch','Fare','Embarked']]
db


# In[4]:


print(db['Sex'].value_counts())
print(db['Age'].value_counts())
print(db['SibSp'].value_counts())
print(db['Parch'].value_counts())
print(db['Fare'].value_counts())
print(db['Embarked'].value_counts())


# In[14]:


db.corr()


# In[6]:


db.info()


# In[7]:


db['Age'].fillna(int(db['Age'].mean()), inplace=True)


# In[17]:


db=db.dropna()
db


# In[18]:


import numpy as np

db['Embarked'] = db['Embarked'].str.replace('S', '1')
db['Embarked'] = db['Embarked'].str.replace('C', '0')
db['Embarked'] = db['Embarked'].str.replace('Q', '2')
db['Embarked']=db['Embarked'].astype(str).astype(int)


# In[20]:



db['Sex'] = db['Sex'].str.replace('female', '1')
db['Sex'] = db['Sex'].str.replace('male', '0')


# In[22]:


db['Sex']=db['Sex'].astype(str).astype(int)
db.corr()


# In[23]:


sns.heatmap(db.corr())


# In[24]:


sns.heatmap(db.corr(),cmap='coolwarm',annot=True)


# In[25]:


db.describe()


# In[62]:


db.groupby(['Embarked']).count()


# In[27]:


db.groupby(['Sex']).count()


# In[63]:


db.groupby(level=0).mean()


# In[28]:


sns.barplot(x='Embarked',y='Sex',data=db)


# In[ ]:


#db['Sex'] = map(lambda x: x.encode('base64','strict'), db['Sex'])
db


# In[ ]:


#df = map(lambda x: x.decode('base64','strict'), db['Sex'])
df


# In[32]:


sns.barplot(x='Fare',y='Embarked',data=db)


# In[33]:


sns.barplot(x='Embarked',y='Fare',data=db)


# In[48]:


sns.barplot(x='Parch',y='Fare',data=db)


# In[34]:


sns.countplot(x='Sex',data=db)


# In[50]:


sns.boxplot(x="Fare", y="Embarked", data=db,palette='rainbow')


# In[41]:


sns.boxplot(data=db,palette='rainbow',orient='h')


# In[42]:


sns.distplot(db['Fare'],kde=False,bins=30)


# In[43]:


sns.jointplot(x='Fare',y='Embarked',data=db,kind='scatter')


# In[44]:


sns.pairplot(db)


# In[45]:


sns.pairplot(db,hue='Embarked',palette='coolwarm')


# In[ ]:


sns.barplot(x='Parch',y='Fare',data=db)


# In[51]:


db.pivot_table(values='Parch',index='Sex',columns=['Embarked'])


# In[47]:


piv = db.pivot_table(values='Fare',index='Sex',columns=['Embarked'])
sns.heatmap(piv)
#most of the men died especially thouth with law fare


# In[52]:


piv = db.pivot_table(values='Parch',index='Sex',columns=['Embarked'])
sns.heatmap(piv)
#most of the welthy women servived , 


# In[56]:



db['Fare'].plot.hist()


# In[57]:



db['Parch'].plot.hist()


# In[58]:



db['Embarked'].plot.hist()


# In[ ]:




