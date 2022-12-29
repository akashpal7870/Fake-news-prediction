#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data_fake=pd.read_csv("C:/Users/palsj/Documents/fakenews.csv")


# In[3]:


data_fake.head()


# In[4]:


data_true=pd.read_csv("C:/Users/palsj/Documents/Truenews.csv")


# In[5]:


data_true.head()


# In[6]:


data_fake["Class"]=0
data_true["Class"]=1


# In[7]:


data_fake.shape,data_true.shape


# In[8]:


data_fake_manual_testing=data_fake.tail(10)
for i in range(23459,23449,-1):
    data_fake.drop([i],axis=0,inplace=True)
    
data_true_manual_testing=data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i],axis=0,inplace=True)
    


# In[9]:


data_fake.shape,data_true.shape


# In[10]:


data_fake_manual_testing['Class']=0
data_true_manual_testing['Class']=1


# In[11]:


data_fake_manual_testing.head(10)


# In[12]:


data_true_manual_testing.head(10)


# In[13]:


data_merge=pd.concat([data_fake,data_true],axis=0)
data_merge.head(10)


# In[14]:


data_merge.columns


# In[15]:


data=data_merge.drop(['title','subject','date'],axis=1)


# In[16]:


data.isnull().sum()


# In[17]:


data=data.sample(frac=1)


# In[18]:


data.head()


# In[19]:


data.reset_index(inplace=True)
data.drop(['index'],axis=1,inplace=True)


# In[20]:


data.columns


# In[21]:


import re
import string


# In[22]:


def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('http ? ://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text


# In[23]:


data['text']=data['text'].apply(wordopt)


# In[24]:


x=data['text']
y=data['Class']


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)


# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)


# In[27]:


LR=LogisticRegression()


# In[28]:


LR.fit(xv_train,y_train)


# In[29]:


pred_lr=LR.predict(xv_test)


# In[30]:


LR.score(xv_test,y_test)


# In[31]:


print(confusion_matrix(y_test,pred_lr))


# In[32]:


print(classification_report(y_test,pred_lr))


# In[33]:


from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred_dt=DT.predict(xv_test)
DT.score(xv_test,y_test)


# In[34]:


print(classification_report(y_test,pred_dt))


# In[35]:


from sklearn.ensemble import GradientBoostingClassifier


# In[36]:


GB=GradientBoostingClassifier(random_state=0)
GB.fit(xv_train,y_train)


# In[37]:


pred_gb=GB.predict(xv_test)


# In[38]:


GB.score(xv_test,y_test)


# In[39]:


print(classification_report(y_test,pred_gb))


# In[40]:


from sklearn.ensemble import RandomForestClassifier


# In[41]:


RF=RandomForestClassifier(random_state=0)
RF.fit(xv_train,y_train)


# In[42]:


pred_rf=RF.predict(xv_test)


# In[43]:


RF.score(xv_test,y_test)


# In[44]:


print(classification_report(y_test,pred_rf))


# In[45]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not a Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    return print("\n\nLR Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),
                                                                                                              output_lable(pred_GB[0]),
                                                                                                              output_lable(pred_RF[0])))
    


# In[46]:


news=str(input())
manual_testing(news)


# In[ ]:





# In[ ]:




