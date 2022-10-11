#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


credits=pd.read_csv(r"/Users/ASUS/Desktop/Project/credits.csv")


# In[5]:


credits


# In[6]:


movies=pd.read_csv(r"/Users/ASUS/Desktop/Project/movies.csv")


# In[7]:


movies


# In[8]:


final=pd.merge(credits,movies)


# In[9]:


final


# In[10]:


final=final[["overview","original_title","id","genres","original_language"]]


# In[11]:


final.isnull().sum()


# In[12]:


final.dropna(inplace=True)


# In[13]:


final.duplicated().value_counts()


# In[14]:


final=final.drop_duplicates(keep="first",inplace=False)


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer(min_df=3,max_features =None, strip_accents= "unicode",analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1,3),stop_words='english')
tfv_matrix=tfv.fit_transform(final["overview"])


# In[16]:


tfv_matrix.shape


# In[17]:


from sklearn.metrics.pairwise import sigmoid_kernel
sig=sigmoid_kernel(tfv_matrix,tfv_matrix)
sig


# In[18]:


index1=pd.Series(final.index,index=final["original_title"]).drop_duplicates()
index1


# In[19]:


def give_recommendation(title,sig=sig):
    idx=index1[title]
    sigmoid_score= list(enumerate(sig[idx]))
    sigmoid_score=sorted(sigmoid_score, key=lambda x:x[1],reverse=True)
    sigmoid_score=sigmoid_score[1:6]
    movie_indices=[i[0] for i in sigmoid_score]
    return final["original_title"].iloc[movie_indices]
give_recommendation("Spectre")


# In[20]:


def give_recommendation(title,sig=sig):
    idx=index1[title]
    sigmoid_score= list(enumerate(sig[idx]))
    sigmoid_score=sorted(sigmoid_score, key=lambda x:x[1],reverse=True)
    sigmoid_score=sigmoid_score[1:6]
    movie_indices=[i[0] for i in sigmoid_score]
    return final["original_title"].iloc[movie_indices]
give_recommendation("Avatar")


# In[24]:


def give_recommendation(title,sig=sig):
    idx=index1[title]
    sigmoid_score= list(enumerate(sig[idx]))
    sigmoid_score=sorted(sigmoid_score, key=lambda x:x[1],reverse=True)
    sigmoid_score=sigmoid_score[1:6]
    movie_indices=[i[0] for i in sigmoid_score]
    return final["original_title"].iloc[movie_indices]
give_recommendation("The Dark Knight Rises")


# In[ ]:




