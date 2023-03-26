#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
from sklearn import linear_model

#Reshaped for Logistic function.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

#predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))
print(predicted)


# In[2]:


import numpy
from sklearn import linear_model

#Reshaped for Logistic function.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

log_odds = logr.coef_
odds = numpy.exp(log_odds)

print(odds)


# In[3]:


import numpy
from sklearn import linear_model

X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

def logit2prob(logr, X):
  log_odds = logr.coef_ * X + logr.intercept_
  odds = numpy.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

print(logit2prob(logr, X))


# In[4]:


import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics 
import seaborn as sn 
import matplotlib.pyplot as plt


# In[5]:


candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690], 
'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7], 
'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5], 
'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1] 
} 
df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted']) 
print (df)


# In[6]:


#dependent and independent
X = df[['gmat', 'gpa','work_experience']] 
y = df['admitted']


# In[7]:


#split train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# In[8]:


print(y_test)


# In[9]:


#performance of logistic regression
logistic_regression= LogisticRegression() 
logistic_regression.fit(X_train,y_train) 
y_pred=logistic_regression.predict(X_test)


# In[10]:


#predicted value
print(y_pred) 


# In[11]:


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'],colnames=['Predicted']) 
sn.heatmap(confusion_matrix, annot=True)


# In[12]:


print('Accuracy: ',metrics.accuracy_score(y_test, y_pred)) 
plt.show()


# In[13]:


#comparing actual and predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
df


# In[15]:


#testing actual and predicted value
def Check(df): 
    if df['Actual']== df['Predicted']: 
        return "True" 
    else: 
        return "False" 
df['Matching'] = df.apply(Check, axis=1) 
df


# In[23]:


#new data to predict
new_candidates = {'gmat': [580,740,680,610,710], 
'gpa': [4.0,3.7,3.3,2.3,3], 
'work_experience': [3,4,6,1,5] 
} 
df2 = pd.DataFrame(new_candidates,columns= ['gmat', 'gpa','work_experience']) 
print(df2)



# In[24]:


#new data prediction
y_pred=logistic_regression.predict(df2)
print(y_pred)


# In[25]:


#plotting the line
x = df["Actual"] 
y = df["Predicted"] 
sn.regplot(x,y,logistic=True,ci=None)


# In[26]:


x = df["Predicted"] 
y = df["Actual"] 
sn.regplot(x,y,logistic=True,ci=None)


# In[ ]:




