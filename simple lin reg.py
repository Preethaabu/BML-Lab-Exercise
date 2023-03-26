#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[41]:


data=pd.read_csv('C:/Users/HP/Downloads/Salary_data.csv')
data.head(5)


# In[42]:


data.info()


# In[43]:


data.describe()


# In[44]:


data.isnull().sum()


# In[45]:


import matplotlib.pyplot as plt
data.hist(bins=5)


# In[46]:


data.corr()


# In[47]:


x=data['YearsExperience'].values.reshape(-1,1)
x


# In[48]:


y=data['Salary'].values.reshape(-1,1)
y


# In[49]:


plt.scatter(x,y,c='r')
plt.show


# In[69]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[70]:


print(x_train)
print("x_train: ",x_train.shape)


# In[71]:


plt.hist(data['Salary'],bins=15)


# In[72]:


lower_limit=data['Salary'].quantile(0.05)
print(lower_limit)


# In[73]:


print(data[data['Salary']<lower_limit])


# In[74]:


upper_limit=data['Salary'].quantile(0.95)
print(upper_limit)


# In[75]:


print(data[data['Salary']>upper_limit])


# In[76]:


data=data[(data['Salary']>lower_limit) & (data['Salary']<upper_limit)]
print(data)


# In[77]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model=lr.fit(x_train,y_train)


# In[78]:


y_train.shape


# In[67]:


x_train.shape


# In[81]:


model.coef_


# In[82]:


model.intercept_


# In[85]:


y_pred=model.predict(x_test)
print(y_pred)


# In[86]:


print(y_test)


# In[87]:


plt.scatter(x,y,c='r')
plt.plot(x_test,y_pred)
plt.xlabel("yoe")
plt.ylabel("salary")


# In[88]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[89]:


print("MAE:",mean_absolute_error(y_test,y_pred))


# In[90]:


print("MSE:",mean_squared_error(y_test,y_pred))


# In[100]:


r2=r2_score(y_test,y_pred)
print(r2)


# In[101]:


import numpy as np
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))


# In[102]:


N=len(y_test)
p=1
ar2=1-((1-r2)*(1-N))/(N-p-1)
print(ar2)


# In[103]:


model.predict([[11.0]])


# In[104]:


score=model.score(x_test,y_test)
print(score)


# In[ ]:




