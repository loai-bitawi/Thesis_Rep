import sys
import json
import os
import numpy as np
from tqdm import  tqdm
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from config import default_settings

class result():
    def __init__(self,dataset,flag):
        self.main_path=default_settings().main_path
        self.rel=json.loads(open(self.main_path+dataset+'/'+'rel.txt').read())
        self.d_list=list(json.loads(open(self.main_path+dataset+'/'+'docs.txt').read()).keys())
        self.res=json.loads(open(self.main_path+dataset+'/'+'results_'+flag+'.txt').read())
        self.queries=json.loads(open(self.main_path+dataset+'/'+'qrs.txt').read())
        
    def crawler(self,q):
        result={}
        for file in (self.res.keys()):
            for doc in self.res[file].keys():
                for k in self.res[file][doc].keys():
                    temp=sum(self.res[file][doc][k][q][1])/len(self.res[file][doc][k][q][1])
                    result[doc]=temp
        return result 

    def scores(self):
        scores=pd.DataFrame(columns=['index','Precision','Recall','F-Score','Top J','Accuracy','indexes']) 
        agg_scores=pd.DataFrame(columns=['index','Precision','Recall','F-Score','Top J','Accuracy','indexes']) 
        
        for j in tqdm(range(1,11),desc='Top J'):
            temp=[]
            ind=[]
            for q in self.queries.keys():
                relevance=self.rel[q]
                result=self.crawler(q)
                sorted_y_pred= sorted(result.items(), key=lambda x: x[1], reverse=False)
                sorted_result=[str(i[0]) for i in sorted_y_pred]
                indexes=[str(sorted_result.index(i)) for i in relevance if i in sorted_result]
                #indexes=','.join(indexes)
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
                tn=len(self.d_list)-tp-fp-fn
                accuracy=(tp+tn)/(tp+tn+fp+fn)
                temp.append([precision,recall,F_score,accuracy])
                ind.append(indexes)
                data_plot=pd.DataFrame(temp,columns=['Precision','Recall','F-Score','Accuracy'])#.agg(['mean'])
                data_plot['Top J']=j
                data_plot['indexes']=ind
                scores=pd.concat((scores,data_plot.reset_index()))
            data_plot2=data_plot[data_plot['Top J']==j].agg(['mean'])
            agg_scores=pd.concat((agg_scores,data_plot2.reset_index()))
        return scores,agg_scores