from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from config import default_settings
import pandas as pd
import matplotlib.pyplot as plt

class mean_of_embedding():
    def __init__(self,dataset):
        self.main_path=default_settings().main_path
        self.docs=json.loads(open(self.main_path+dataset+'/'+'docs.txt').read())
        self.qrs=json.loads(open(self.main_path+dataset+'/'+'qrs.txt').read())
        self.special_char = '@ _ ! # $ % ^ & * ( ) < > ? / \ | } { ~ : ; [ ] - . , '
        self.special_char = self.special_char.split()
        self.special_char.extend(['[CLS]','[SEP]','[UNK]'])
        self.rel=json.loads(open(self.main_path+dataset+'/'+'rel.txt').read())

    
    def retreival(self,source,key):
        words=[]
        vecs=[]
        for word in source[key]:
            if word == '[CLS]':
                continue
            elif word == '[SEP]':
                continue
            elif word =='[UNK]':
                continue
            if word not in words:
                words.append(word)
                vecs.append(source[key][word])
        return words, np.array(vecs)
    
    
    def kthSmallest(self,arr, k): 
        ar=np.sort(arr)
        return ar[k-1]

     
    def run(self,doc):
        d_vecs=np.array(doc[1])
        d=len(d_vecs)
        doc_mean=np.mean(d_vecs,axis=0)        
        distances={}
        for q in self.qrs.keys():
            q_vecs=np.array(self.qrs[q][1])
            q_words=self.qrs[q][0]
            qr_mean=np.mean(q_vecs,axis=0)
            dist=pdist([doc_mean,qr_mean],metric='euclidean')
            dist_cos=pdist([doc_mean,qr_mean],metric='cosine')
            distances[q]=[dist[0], dist_cos[0]]
        return distances

    def runner(self):
        self.results={}
        for d in tqdm(self.docs.keys()):
            self.results[d]=self.run(self.docs[d])


    def crawler(self,q,flag):
        result={}
        for doc in (self.results.keys()):            
            temp=self.results[doc][q][flag]
            result[doc]=temp
        return result 

    def results(self,flag='Euclidean'):
        self.runner()    
        scores=pd.DataFrame(columns=['index','Precision','Recall','F-Score','Top J','Accuracy','indexes']) 
        agg_scores=pd.DataFrame(columns=['index','Precision','Recall','F-Score','Top J','Accuracy','indexes']) 
        if flag=='Euclidean': flag=0
        elif flag=='Cosine': flag=1

        for j in tqdm(range(1,11),desc='Top J'):
            for q in self.qrs.keys():
                temp=[]
                ind=[]
                index=[]
                relevance=self.rel[q]
                result=self.crawler(q,flag) # 0 for euclidean , 1 for Cosine
                sorted_y_pred= sorted(result.items(), key=lambda x: x[1], reverse=False)
                sorted_result=[str(i[0]) for i in sorted_y_pred]
                indexes=[str(sorted_result.index(i)) for i in relevance if i in sorted_result]
                y_pred=[str(i[0]) for i in sorted_y_pred[:j]]
                y_true=[1 for i in y_pred if i in relevance]   
                if len(y_pred)==0: precision=0
                else: precision= len(y_true)/len(y_pred)
                # precision= len(y_true)/len(y_pred)
                recall=len(y_true)/min(len(relevance),j)
                if precision+recall>0:
                    F_score=2*(precision*recall)/(precision+recall)
                else: F_score=0
                tp=len(y_true)
                fp=len(y_pred)-tp
                fn=min(len(relevance),j)-tp
                tn=len(self.docs)-tp-fp-fn
                accuracy=(tp+tn)/(tp+tn+fp+fn)
                temp.append([precision,recall,F_score,accuracy])
                ind.append(indexes)
                index.append(q)
                data_plot=pd.DataFrame(temp,columns=['Precision','Recall','F-Score','Accuracy'])#.agg(['mean'])
                data_plot['Top J']=j
                data_plot['indexes']=ind
                data_plot['index']=index
                scores=pd.concat((scores,data_plot))
            agg=scores.agg(['mean'])
            agg['Top J']=j
            agg_scores=pd.concat((agg,agg_scores))
        return scores, agg_scores
