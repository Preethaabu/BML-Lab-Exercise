#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd


# In[9]:


data=pd.read_csv('C:/Users/HP/Downloads/weight-height.csv')
data


# In[10]:


data.isnull().sum()


# In[11]:


male=data[data['Gender']=="Male"]
male


# In[12]:


female=data[data['Gender']=='Female']
female


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[14]:


fig,ax=plt.subplots(1,2,figsize=(12,6))

ax1=sns.distplot(male['Height'],ax=ax[0],label="mh")
ax1=sns.distplot(female['Height'],ax=ax[0],label="fh")
ax1.set_title("hd")
ax1.legend

ax2=sns.distplot(male['Weight'],ax=ax[1],label="mh")
ax2=sns.distplot(female['Weight'],ax=ax[1],label="fh")
ax2.set_title("hd")
ax2.legend


# In[16]:


fig,ax=plt.subplots(1,2,figsize=(12,6))
ax1=sns.boxplot(x="Gender",y="Weight",data=data,ax=ax[0])
ax2=sns.boxplot(x="Gender",y="Height",data=data,ax=ax[1])


# In[19]:


from sklearn.model_selection import train_test_split
X=data.drop(["Gender"],axis=1)# axis=1 we drop a full gender column
Y=data["Gender"]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.30,random_state=0)


# In[20]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model=model.fit(X_train, y_train)


# In[21]:


y_pred=model.predict(X_test)
y_pred


# In[23]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
accuracy_score(y_test, y_pred)


# In[25]:


confusion_matrix(y_test, y_pred)
confusion_matrix


# In[26]:


print(classification_report(y_test,y_pred))
classification_report


# In[28]:


score=model.score(X_test,y_test)
score


# In[29]:


data.describe()


# In[ ]:




