#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv('hearts.csv')
data


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[8]:


y=data['target']
x=data.drop('target',axis=1)
print(x.head())
print(y.head())


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
scale=scaler.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)


# In[10]:


model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[11]:


score=accuracy_score(y_test,y_pred)
score


# In[12]:


confusion_matrix(y_test,y_pred)


# In[16]:


tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()
(tn,fp,fn,tp)


# In[ ]:





# In[17]:


report=classification_report(y_test,y_pred)
print("Classification report:",report)


# In[ ]:




