#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
students=["pre","leka","anu","yo","nans"]
values=[16,17,54,34,17]
fig=plt.figure(figsize=(10,7))
plt.pie(values,labels=students)
plt.show()


# In[4]:


students=["tiger","lion","zebra","fox"]
values=[12,45,34,26]
exp=[0.2,0,0,0]
plt.pie(values,labels=students,startangle=90,explode=exp)
plt.show()


# In[5]:


flowers=["rose","hibiscus","lilly","lotus"]
values=[15,25,12,8]
plt.pie(values,labels=flowers,explode=exp,shadow=True,autopct='%2.1f%%')


# In[7]:


food=["pizza","cake","burger"]
values=[12,85,47]
exp=[0.2,0,0]
plt.pie(values,labels=students,startangle=90,explode=exp,shadow=True,autopct='%2.1f%%',colors=['b','g','r','y'])
plt.legend(title="Food")
plt.show()


# In[8]:


fig=plt.figure(figsize=(10,5))
students=["pre","leka","yo","nans","anu"]
marks=[56,67,55,78,56]
plt.bar(students,marks,color='green',width=0.5)
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Marks secured by student")
plt.show()


# In[10]:


import pandas as pd
plotdata=pd.DataFrame({"2018":[34,56,47],
                      "2020":[63,23,45],
                      "2022":[57,46,83]},
                     index=['pre','leka','anu'])
plotdata.plot(kind="bar",stacked=True,figsize=(12,6))
plt.title("Marks")
plt.xlabel("students")
plt.ylabel("marks")


# In[13]:


plotdata=pd.DataFrame({"2019":[23,45,76],
                      "2021":[24,54,67],
                      "2022":[23,56,43]},
                     index=['pre','leka','nans'])
plotdata.plot(kind="bar",figsize=(12,6),color=['b','g','r'])
plt.title("Marks secured by students")
plt.xlabel("Students")
plt.ylabel("Marks")


# In[15]:


fig=plt.figure(figsize=(10,5))
students=["pre","leka","yo","nans","anu"]
marks=[56,67,55,78,56]
plt.barh(students,marks,color='green')
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Marks secured by student")
plt.show()


# In[ ]:




