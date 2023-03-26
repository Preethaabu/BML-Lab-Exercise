#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("50_Startups.csv") 
data.head()


# In[3]:


from sklearn import preprocessing 
#creating labelEncoder 
le = preprocessing.LabelEncoder() 
# Converting string labels into numbers. 
state_encoded=le.fit_transform(data['State']) 
data['State'] = state_encoded 
data.head()


# In[5]:


data.info()
data.isnull().sum()


# In[6]:


data.describe()


# In[7]:


X = data.iloc[:,:4] 
X.head()


# In[12]:


Y = data.iloc[:,4]
Y.head()


# In[13]:


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state=355) 
from sklearn.linear_model import LinearRegression 
mlr = LinearRegression() 
model=mlr.fit(x_train,y_train)


# In[14]:


y_predict = model.predict(x_test) 
y_predict


# In[15]:


plt.scatter(y_test,y_predict) 
plt.xlabel('y_test', fontsize=18) 
plt.ylabel('y_pred', fontsize=16)


# In[24]:


from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_predict) 
meanSqErr = metrics.mean_squared_error(y_test, y_predict) 
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_predict)) 
print('R squared: {:.2f}'.format(mlr.score(X,Y)*100)) 
print('Mean Absolute Error:', meanAbErr) 
print('Mean Square Error:', meanSqErr) 
print('Root Mean Square Error:', rootMeanSqErr)


# In[16]:


print(model.score(x_test, y_test))


# In[ ]:




