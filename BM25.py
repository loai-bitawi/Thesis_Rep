#python -m spacy download en_core_web_sm

import spacy
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import json
from data_parser import data_parser
from config import default_settings

class BM25():
    def __init__(self,dataset):
        self.main_path=default_settings().main_path
        self.dataset=dataset
        self.rel=json.loads(open(self.main_path+dataset+'/'+'rel.txt').read())
        self.nlp = spacy.load("en_core_web_sm")
        self.docs=json.loads(open(self.main_path+dataset+'/'+'docs.txt').read())
        self.docs=self.docs['dict']
        self.docs=[str(i) for i in list(self.docs.keys())]
        self.d_list=self.docs
        self.all_docs,self.queries,_= data_parser().parser(dataset)
        self.all_docs={str(i):self.all_docs[i] for i in self.all_docs.keys()}
        self.q_list=list(self.queries.keys())
        self.all_docs={str(i):self.all_docs[i] for i in self.docs}
        self.docs=[i.lower() for i in self.all_docs.values()]
        self.queries=[i.lower() for i in list(self.queries.values())]
        self.tok_text=[] 
        for doc in tqdm(self.nlp.pipe(self.docs)):
           tok = [t.text for t in doc if t.is_alpha]
           self.tok_text.append(tok)        
        self.tok_queries=[]
        for doc in tqdm(self.nlp.pipe(self.queries)):
           tok = [t.text for t in doc if t.is_alpha]
           self.tok_queries.append(tok)
        self.bm25 = BM25Okapi(self.tok_text)
           
    def run(self,topj):
        ranked_docs={}
        doc_list=[]
        for tk,q in enumerate(self.tok_queries):
            answers = self.bm25.get_top_n(q, self.docs, n=topj)
            doc_id=[str(self.d_list[self.docs.index(i)]) for i in answers] # actual results
            doc_list.extend(doc_id)
            ranked_docs[tk]=doc_id
            
        return ranked_docs,list(set(doc_list))
