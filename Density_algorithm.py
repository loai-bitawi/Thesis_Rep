
import multiprocessing

from joblib import Parallel, delayed, parallel_backend

import sys
import concurrent.futures
from queue import Queue
import json
import os
import numpy as np
from tqdm import  tqdm
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
from Distance_Matrix import distances
import math
import re
from config import default_settings
from sklearn.feature_extraction.text import TfidfVectorizer
from BM25 import BM25

class density():
    def __init__(self,flag,dataset,filename):
        self.dataset=dataset
        self.main_path=default_settings().main_path
        self.special_char = '@ _ ! # $ % ^ & * ( ) < > ? / \ | } { ~ : ; [ ] - . , '
        self.special_char = self.special_char.split()
        self.special_char.extend(['[CLS]','[SEP]','[UNK]'])
        self.flag=flag
        self.docs=json.loads(open(self.main_path+dataset+'/'+filename).read())
        self.queries=json.loads(open(self.main_path+dataset+'/'+'qrs.txt').read())
        sub=os.listdir(self.main_path+self.dataset+'/')        
        files=[i for i in sub if len(re.findall('\d\w\w*.npy', i))>0] 
        if len(files)==0:
            distances(self.main_path+self.dataset+'/',filename).runner(self.flag)
        _ , self.q_lens =self.query_vecs()
        self.TFIDF()
        self.BM25_ranker=BM25(self.dataset)
        
    def retreival(self,source,key):
           words=[]
           vecs=[]
           for i, word in enumerate(source[key][0]): 
               if word not in ENGLISH_STOP_WORDS and word not in self.special_char and word not in ["'",'"'] and '#' not in word:
                   words.append(word)
                   vecs.append(source[key][1][i])
           return words, np.array(vecs)
    
    def query_vecs(self):
        vecs=[]
        l=[]
        for q in tqdm(self.queries.keys()):
            _ , q_vecs=self.retreival(self.queries,q)
            if len(vecs): vecs=np.concatenate((vecs,q_vecs))
            else: vecs=q_vecs
            l.append(len(q_vecs))
        return vecs ,l
    
    def kthSmallest(self,arr,k): 
        ar=np.sort(arr)
        return ar[k-1]

    def TFIDF(self):
        docs_list=[]
        for i in self.docs.keys():
            docs_list.append(" ".join(self.docs[i][0]))
        doc_tk=json.loads(open(self.main_path+self.dataset+'/'+'doc_tokens_all.txt').read())
        vectorizer = TfidfVectorizer(token_pattern='\w\w*')
        vectors = vectorizer.fit_transform(docs_list)
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        self.tf_idf = pd.DataFrame(denselist, columns=feature_names,index=list(doc_tk.keys()))
        self.tf_idf.to_csv(self.main_path+self.dataset+'/'+'TFIDF.csv')
        self.tf_idf=pd.read_csv(self.main_path+self.dataset+'/'+'TFIDF.csv') 
        self.tf_idf.index=self.tf_idf['Unnamed: 0'].astype(str)

    def run(self,matt):
        m=matt[1]
        doc_name=matt[0]
        d=len(m)-sum(self.q_lens)
        k_match={}
        if d>=10: it=10
        else: it=d
        k=d
        q_lofs={}
        for q_i,q in enumerate(self.queries.keys()):
            q_words,_=self.retreival(self.queries, q)
            indexes=list(range(0,d))
            start=d+sum(self.q_lens[:q_i])
            end=start+self.q_lens[q_i]
            indexes.extend(range(start,end))
            others=[z for z in range(len(m)) if z not in indexes]
            mat=np.concatenate((m[:d],m[start:end]),axis=0)
            mat=np.delete(mat,others,1)    
            match_qr=[]
            words=[]
            for l in (range(d,len(mat))):
                ind=list(range(0,d))
                ind.append(l)
                kth_nb=[self.kthSmallest(mat[w][ind],k+1) for w in ind]  
                lrd=[]               
                for i in ind:
                    rd=[]
                    for count, j in enumerate(ind):
                        if i==j:
                            continue
                        else:   
                            r_distances=np.maximum(kth_nb[count],mat[i][j])
                            rd.append(r_distances)
                    if len(rd)>0:
                        lrd.append(len(rd)/sum(rd))
                lof=((sum(lrd)-lrd[-1])/(len(lrd)-1))/lrd[-1]
                if q_words[l-d] in self.tf_idf.columns:
                    if self.tf_idf.loc[doc_name][q_words[l-d]]>0:
                        ##Used to test the normalization effect 
                        #match_qr.append(lof/tf_idf.loc[int(doc_name)][q_words[l-d]]*(1+math.log(len(docs[doc_name]),10))) 
                        match_qr.append(lof/self.tf_idf.loc[doc_name][q_words[l-d]])    
                    else: 
                        ##Used to test the normalization effect 
                        #match_qr.append(lof/0.00001*(1+math.log(len(docs[doc_name]),10)))
                        match_qr.append(lof/0.00001)
                else:  
                    ##Used to test the normalization effect 
                    #match_qr.append(lof*(1+math.log(len(docs[doc_name]),10)))
                    match_qr.append(lof)
                words.append(q_words[l-d])
            q_lofs[q]=[words,match_qr]
        k_match[k]=q_lofs   
        return {doc_name:k_match}
    
    def mp_run(self,mat,job_type='Multithreading',njobs=2):
        if job_type=='Multithreading':
            processed_list=[]
            start = time.process_time()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            with concurrent.futures.ThreadPoolExecutor() as executer:    
                    results=list(tqdm(executer.map(self.run,mat), total=len(mat)))
                    processed_list.extend(results)
            end = time.process_time()
            print('Elapsed time for is:', str(end - start),'Seconds')            
            
        elif job_type=='Parallelism':
            start = time.process_time()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            num_cores = multiprocessing.cpu_count()
            with parallel_backend("loky", inner_max_num_threads=2):
                processed_list = list(tqdm(Parallel(n_jobs=njobs)(delayed(self.run)(i) for i in mat)))
            end = time.process_time()
            print('Elapsed time ',' is:', str(end - start),'Seconds')
        return processed_list

    def ranker(self,mat,keys,topj):
        self.ranked_docs,self.ranked_list=self.BM25_ranker.run(topj)
        ranked_keys=[i for i in keys if i in self.ranked_list]
        ranked_mat=[mat[keys.index(i)] for i in ranked_keys] 
        ranked_set=list(zip(ranked_keys,ranked_mat))
        return ranked_set
        
    def runner(self,job_type='Multithreading',njobs=2,topj=10):
        results={}
        start = time.process_time()
        chunks=json.loads(open(self.main_path+self.dataset+'/'+'chunks_'+self.flag+'.txt').read())
        sub=os.listdir(self.main_path+self.dataset+'/')        
        files=[i for i in sub if len(re.findall('\d\w\w*.npy', i))>0]
        for file in tqdm(files,desc='files'): 
            mat=np.load(self.main_path+self.dataset+'/'+file,allow_pickle=True)
            keys=chunks[self.main_path+self.dataset+'/'+file]
            sub_list=self.ranker(mat,keys,topj)
            # mat=list(zip(keys,mat))
            res=self.mp_run(sub_list,job_type,njobs)
            results[file]={list(i.keys())[0]:i[list(i.keys())[0]] for i in res}
            res=None
            mat=None
        end = time.process_time()
        print('Elapsed time:', str(end - start),'Seconds')  
        with open(self.main_path+self.dataset+'/'+'results_'+self.flag+'.txt', 'w') as file:
             file.write(json.dumps(results)) 


