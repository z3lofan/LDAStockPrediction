#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import re
import pandas as pd

results=[]

for filename in os.listdir('.'):
    if filename.startswith('Results_'):
        #print (filename.split('_'))
        _,n_topics,n_words,day_offset,merge_by_day=filename.split('_')
        merge_by_day=merge_by_day.startswith('True')
        print(n_topics,n_words,day_offset,merge_by_day)
        
        df=pd.read_csv(filename,sep=';')
        results.append({'n_topics':n_topics,'n_words':n_words,'day_offset':day_offset,'merge_by_day':merge_by_day,'df_results':df})
        
    


# In[25]:


xx=[]
yy=[]

for result in results:
    df= result['df_results']
    #print(df)
    #print(df[df['test_precision']= df['test_precision'].max()])
    best_row=df.iloc[df['test_precision'].argmax()]
    xx.append(result['n_topics'])
    yy.append(best_row['test_precision'])
    print(result['n_words'],result['n_topics'],result['merge_by_day'],best_row['test_precision'])
import matplotlib.pyplot as plt

plt.xlabel('topics')
plt.ylabel('precision')
plt.scatter(xx,yy)
    
    

