import time 
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform
import multiprocessing
from joblib import Parallel, delayed, parallel_backend
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import concurrent.futures


class distances():
    def __init__(self,main_path,file_name,dataset,stop_words='True',subfile=''):
        self.subfile=subfile
        self.stop_words=stop_words
        self.file_name= file_name
        self.main_path=main_path
        self.queries=json.loads(open(main_path+'qrs.txt').read())
        self.queries={'4':self.queries['4']}      #########################################
        self.special_char = '@ _ ! # $ % ^ & * ( ) < > ? / \ | } { ~ : ; [ ] - . , '
        self.special_char = self.special_char.split()
        self.special_char.extend(['[CLS]','[SEP]','[UNK]'])
        self.q_vecs, self.q_lens =self.query_vecs()
        self.flag='Euclidean'
        self.docs=dataset

    def retreival(self,source,key):
        words=[]
        vecs=[]
        for i, word in enumerate(source[key][0]):
            if self.stop_words:
                if word not in ENGLISH_STOP_WORDS and word not in self.special_char and word not in ["'",'"'] and '#' not in word:
                    words.append(word)
                    vecs.append(source[key][1][i])
            else:
                words.append(word)
                vecs.append(source[key][1][i])
        return words, np.array(vecs)
    
    def query_vecs(self):
        vecs=[]
        l=[]
        for q in tqdm(self.queries.keys()):
            _ , q_vecs=self.retreival(self.queries, q)
            if len(vecs): vecs=np.concatenate((vecs,q_vecs))
            else: vecs=q_vecs
            l.append(len(q_vecs))
        return vecs ,l    
    
    def mat_calc(self,doc):
        _, d_vecs=self.retreival(self.docs, doc)
        d_matrix=squareform(pdist(np.concatenate((d_vecs,self.q_vecs)),metric=self.flag))
        return d_matrix
        
    def run_dist(self,it,job_type,njobs):   
        if job_type=='Multithreading':
            results=[]
            start = time.process_time()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            with concurrent.futures.ThreadPoolExecutor() as executer:    
                    res=list(tqdm(executer.map(self.mat_calc,it), total=len(it)))
                    results.extend(res)
            end = time.process_time()
            print('Elapsed time for is:', str(end - start),'Seconds')                        
            
        elif job_type=='Parallelism':
            start = time.process_time() 
            num_cores = multiprocessing.cpu_count()
            with parallel_backend(backend='loky'):
                results = list(tqdm(Parallel(n_jobs=njobs)(delayed(self.mat_calc)(i) for i in it)))          
            end = time.process_time()
            print('Elapsed time for data import is:', str(end - start),'Seconds')
            
        return results

    def runner(self,flag,job_type='Multithreading',njobs=2):
        self.flag= flag
        self.doc_list=list(self.docs.keys())
        self.chunks=[self.doc_list[i:i + 100] for i in range(0, len(self.doc_list), 100)]
        doc_mats=None
        i=0
        for chunk in tqdm(self.chunks):
            doc_mats=self.run_dist(chunk,job_type,njobs)
            np.save(self.main_path+'/'+self.subfile+'/'+str(i)+self.flag+".npy",doc_mats)
            doc_mats=None
            i+=1        
        ch={}
        i=0
        for chunk in tqdm(self.chunks):
            ch[self.main_path+self.subfile+'/'+str(i)+self.flag+".npy"]=chunk
            i+=1
        with open(self.main_path+'/'+self.subfile+'/'+"chunks_"+self.flag+".txt", "w") as file:
             file.write(json.dumps(ch)) 
             
# x=distances('E:/GitHub/Thesis/lisa/','docs.txt')
# x.runner('Euclidean')