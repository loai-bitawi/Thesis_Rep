# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:23:50 2022

@author: Loai
"""

import os
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
special_char = '@ _ ! # $ % ^ & * ( ) < > ? / \ | } { ~ : ; [ ] - . , '
special_char = special_char.split()
special_char.extend(['[CLS]','[SEP]','[UNK]'])
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

docs=json.loads(open('E:/GitHub/Thesis/Lisa/doc_tokens_all.txt').read())
#files=[i for i in sub if len(re.findall('TFIDF.csv', i))>0] 
docs={str(i):[j for j in docs[i] if j not in ENGLISH_STOP_WORDS and j not in special_char and j not in ["'",'"'] and '#' not in j] for i in docs.keys()} ###########

docs_list=[]
for i in docs.keys():
    docs_list.append(" ".join(docs[i]))


vectorizer = TfidfVectorizer(token_pattern='\w\w*')
vectors = vectorizer.fit_transform(docs_list)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
tf_idf = pd.DataFrame(denselist, columns=feature_names,index=list(docs.keys()))

tf_idf.to_csv('E:/GitHub/Thesis/Lisa/'+'TFIDF.csv')
tf_idf=pd.read_csv('E:/GitHub/Thesis/Lisa/'+'TFIDF.csv') 
tf_idf.index=tf_idf['Unnamed: 0'].astype(str)