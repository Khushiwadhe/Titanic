#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


titanic_data = pd.read_csv("Titanic-Dataset.csv") 
titanic_data


# In[3]:


titanic_data.head()


# In[4]:


titanic_data.info()


# In[5]:


titanic_data.describe()


# In[6]:


titanic_data.isnull().sum()


# In[8]:


# we will fill blank with median value
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)


# In[9]:


#Count the Embarked
titanic_data['Embarked'].value_counts()


# In[10]:


# replace blanks with mode value
titanic_data['Embarked'].fillna('S', inplace=True)


# In[11]:


# check Null value in data
titanic_data.isnull().sum()


# In[19]:


# In fare column has also null value, replace with median
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)


# In[20]:


# We will remove the "Cabin" column because it contains a significant number of missing values.
titanic_data.drop(columns="Cabin", inplace=True)


# In[21]:


#Last check null value and Dataset
print(titanic_data.isnull().sum())
print(titanic_data.head())


# In[22]:


import seaborn as sns


# In[23]:


titanic_data['Survived'].value_counts()


# In[24]:


sns.countplot(data=titanic_data, x= 'Survived')


# In[25]:


sns.countplot(data=titanic_data,x='Pclass')


# In[26]:


sns.countplot(data=titanic_data,x='Sex')


# In[28]:


sns.histplot( data=titanic_data,x='Age')
plt.show()


# In[29]:


start coding or generate with AI


# In[31]:


sns.countplot(x=titanic_data['Survived'], hue=titanic_data['Pclass'])
plt.show()


# In[32]:


titanic_data['Sex'].head()


# In[33]:


sns.countplot(x=titanic_data['Survived'], hue=titanic_data['Sex'])
plt.show()


# In[34]:


sns.histplot(x=titanic_data['Age'], hue=titanic_data['Survived'], multiple='stack')
plt.show()


# In[35]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
titanic_data['Sex'] = labelencoder.fit_transform(titanic_data['Sex'])
titanic_data.head()


# In[36]:


sns.countplot(x=titanic_data['Sex'], hue=titanic_data['Survived'])
plt.show()


# In[38]:


titanic_data.drop(columns=["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Fare", "Age", "Embarked"], inplace=True)


# In[39]:


titanic_data.head()


# In[40]:


X=titanic_data[['Sex', 'Pclass']]
Y=titanic_data['Survived']


# In[42]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix
log= LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)


# In[46]:


pred =log.predict(X_test)
pred


# In[51]:


print("Accuracy score:", accuracy_score(Y_test, pred))
print("Matrix", confusion_matrix (Y_test,pred))


# In[52]:


Y_test


# In[65]:


submission=x.iloc[:,:].values
y_final=log.predict(submission)


# In[66]:


y_final.shape


# In[61]:


final =pd.DataFrame() 
final["Sex"]= X['Sex']
final["survived"]=y_final


# In[62]:


final.to_csv("submission.csv",index=False)


# In[64]:


import warnings
warnings.filterwarnings ("ignore")
result = log.predict([[5,6]])
if(result == 0): print("So sory, Not Survived")
else: print("Survived")


# In[ ]:




