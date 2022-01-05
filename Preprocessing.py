
from data_parser import data_parser
from tqdm import tqdm 
from transformers import BertTokenizer
from transformers import BertModel
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from config import default_settings
import random

class preprocessing():
    def __init__(self,model_name="bert-base-uncased"):
        self.parser=data_parser()
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.defaults=default_settings()
    
    def data_save(self,main_path,doc_words, words,doc_vecs, vec_qr,rel_ass):
        with open(main_path+'doc_words.txt', 'w') as file:
             file.write(json.dumps(doc_words))        
        with open(main_path+'doc_tokens_all.txt', 'w') as file:
             file.write(json.dumps(words))
        with open(main_path+'docs.txt', 'w') as file:
              file.write(json.dumps(doc_vecs))
        with open(main_path+'qrs.txt', 'w') as file:
             file.write(json.dumps(vec_qr))
        with open(main_path+'rel.txt', 'w') as file:
              file.write(json.dumps(rel_ass))

    def embedder(self,docs,queries,rel_ass):
        vec_docs={}
        for key in tqdm(docs.keys()):   
            encoding=self.tokenizer.encode_plus(docs[key],add_special_tokens=True,max_length=512, truncation=True,return_tensors='pt') 
            tokens=self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            outputs = self.model(**encoding)
            vecs = outputs.last_hidden_state.detach().numpy()
            t_vecs=[]
            t_words=[]
            for tk in range(vecs.shape[1]): 
                t_vecs.append(vecs[0][tk].tolist())
                t_words.append(tokens[tk])
            vec_docs[key]=[t_words,t_vecs]
            if len(vec_docs)==10:
                break

        vec_qr={}
        for key in tqdm(queries.keys()):
            encoding=self.tokenizer.encode_plus(queries[key],max_length=512,add_special_tokens=False, return_token_type_ids=False, padding=False, return_attention_mask=True, return_tensors='pt')
            tokens=self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            outputs = self.model(**encoding)
            vecs_q = outputs.last_hidden_state.detach().numpy()
            t_vecs=[]
            t_words=[]
            for tk in range(len(vecs_q[0])):
                t_vecs.append(vecs_q[0][tk].tolist())
                t_words.append(tokens[tk])
            vec_qr[key]=[t_words,t_vecs]

            
        docs=vec_docs
        vecs={i:docs[i][1] for i in docs.keys() } 
        words={i:docs[i][0] for i in docs.keys()} 
        doc_vecs={}
        doc_words={}
        #save output in text files
        special_char = '@ _ ! # $ % ^ & * ( ) < > ? / \ | } { ~ : ; [ ] - . , '
        special_char = special_char.split()
        special_char.extend(['[CLS]','[SEP]','[UNK]'])
        for key in docs.keys():
            temp_v=[]
            temp_w=[]
            for i, word in enumerate(words[key]):
                if word not in ENGLISH_STOP_WORDS and word not in special_char and word not in ["'",'"'] and '#' not in word:
                    temp_v.append(vecs[key][i])
                    temp_w.append(word)
            doc_vecs[key]=[temp_w,temp_v]
            doc_words[key]=temp_w

        return doc_words, words,doc_vecs, vec_qr

    def sampler(self,docs,rel,doc_vecs,main_path,size=-1):
            q_sample={}
            sample_docs=[]
            sample=[]
            all_rel=[]
            for a in rel.keys():
                all_rel.extend(rel[a])  
            all_rel=list(set(all_rel))
            for a in rel.keys():
                doc_list=[i for i in docs.keys() if i not in rel[a] and i not in sample_docs and i  not in all_rel]
                if size==-1:
                    print(a,len(rel[a]),len(doc_list))
                    sample=random.sample(doc_list,len(rel[a]))
                else:
                    sample= random.sample(doc_list,size)  
                q_sample[a]=[sample,rel[a]]
                sample_docs.extend(sample)
                sample_docs.extend(rel[a])
            sample_docs=list(set(sample_docs))
            sample_docs={d:doc_vecs[d] for d in sample_docs}  
            with open(main_path+'sample_docs.txt', 'w') as file:
                  file.write(json.dumps(sample_docs)) 
    
    def lisa_preprocessing(self,sample=False,sample_size=-1):
        main_path=self.defaults.main_path+'Lisa/'
        docs,queries,rel_ass= self.parser.lisa_parser()    
        doc_words, words,doc_vecs, vec_qr= self.embedder(docs,queries,rel_ass)
        self.data_save(main_path,doc_words, words,doc_vecs, vec_qr,rel_ass)
        if sample:
            self.sampler(docs,rel_ass,doc_vecs,main_path,size=sample_size)

    def aila_preprocessing(self,sample=False,sample_size=-1):
        main_path=self.defaults.main_path+'Aila/'
        docs,queries,rel_ass= self.parser.aila_parser()    
        doc_words, words,doc_vecs, vec_qr= self.embedder(docs,queries,rel_ass)
        self.data_save(main_path,doc_words, words,doc_vecs, vec_qr,rel_ass)
        if sample:
            self.sampler(docs,rel_ass,doc_vecs,main_path,size=sample_size)
    def cran_preprocessing(self,sample=False,sample_size=-1):
        main_path=self.defaults.main_path+'Cranfield/'
        docs,queries,rel_ass= self.parser.cran_parser()
        doc_words, words,doc_vecs, vec_qr= self.embedder(docs,queries,rel_ass)
        self.data_save(main_path,doc_words, words,doc_vecs, vec_qr,rel_ass)
        if sample:
            self.sampler(docs,rel_ass,doc_vecs,main_path,size=sample_size)
            
    

