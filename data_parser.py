from tqdm import  tqdm
import os
import re
from config import default_settings
class data_parser():
    def __init__(self):
        self.defaults=default_settings()
    
    def parser(self,dataset):
        if dataset.lower()=='lisa':
            docs,queries, rel_ass=self.lisa_parser()
        elif dataset.lower()=='cranfield':
            docs,queries, rel_ass=self.cran_parser()
        elif dataset.lower()=='aila':
            docs,queries, rel_ass=self.aila_parser()
        
        return docs,queries, rel_ass
            
    def aila_parser(self):
        main_path=self.defaults.main_path+'Aila/data/'
        #-----------------------------------------------------------
        #Queries 
        qrs=open(main_path+'Query_doc.txt').readlines()
        qrs=[i.split('||')for i in qrs] 
        queries={i[0]:i[1] for i in qrs}                    
        #-----------------------------------------------------------
        #relevance assessment
        temp=open(main_path+'relevance_judgments_priorcases.txt').readlines()
        temp=[item.split() for item in temp]
        rel_ass={}
        for i in temp:
            if i[0] in rel_ass.keys():
                if int(i[3])==1:
                    rel_ass[i[0]].append(i[2])
            else:
                if int(i[3])==1:
                    rel_ass[i[0]]=[i[2]]
        #-----------------------------------------------------------
        #Documents
        docs={}
        sub_folder=main_path+'Object_casedocs'
        for file in tqdm(os.listdir(sub_folder)):
            filename = sub_folder+'/'+file 
            with open(filename, 'r') as content:
                data=content.read()
            name=re.sub('.txt','',file)
            docs[name]=data                    
        return docs,queries, rel_ass
    
    def lisa_parser(self):
            main_path=self.defaults.main_path+'lisa/data/'
            sub=os.listdir(main_path)
            #-----------------------------------------------------------
            #Queries 
            qrs=open(main_path+'/'+'LISA.QUE').read().split('#')
            queries={}
            for item in tqdm(qrs): 
                splitted=re.split('\n', item, flags=re.MULTILINE)
                if len(splitted[0])==0:
                    del splitted[0]
                if len(splitted[0]):
                    queries[int(splitted[0])]=" ".join(splitted[1:])
            #-----------------------------------------------------------
            #relevance assessment
            rel=open(main_path+'/'+'LISARJ.NUM').readlines()
            rel=[item.split() for item in rel]
            rel_ass={}
            j=1
            for i in tqdm(range(len(rel))):
                if int(rel[i][0])==j:
                    rel_ass[int(rel[i][0])]=[int(item) for item in rel[i][2:]]
                    j+=1
                else:
                    rel_ass[j-1].extend([int(item) for item in rel[i]])    
            #-----------------------------------------------------------
            #Documents
            sub.remove('LISA.QUE')
            sub.remove('LISA.REL')
            sub.remove('LISARJ.NUM')
            sub.remove('README')
            docs={}
            for file in tqdm(sub):
                filename = main_path+'/'+file 
                with open(filename, 'r') as content:
                    data=content.read().split('********************************************')
                    for doc in data:
                        splitted=doc.split('\n')
                        if len(splitted[0])==0:
                            del splitted[0]
                        doc_num=re.sub('Document +', '', splitted[0])
                        splitted=[re.sub('  +','',i) for i in splitted]
                        if len(doc_num)==0:
                            break
                        title="".join(splitted[1:splitted.index('')])
                        text=" ".join(splitted[splitted.index('')+1:])
                        #text=" ".join(splitted)
                        docs[int(doc_num)]=text
            return docs,queries, rel_ass

    def cran_parser(self):       
        main_path=self.defaults.main_path+'Cranfield/data/'
        sub=os.listdir(main_path)
        #-----------------------------------------------------------
        #Queries 
        qrs=open(main_path+'/'+'cran.Qry').readlines()
        queries={}
        start=0
        num=0
        for i,item in enumerate(qrs):
            if len(re.findall('.I \d\d*',item)):
                if start>0:
                    queries[num]=' '.join([i.strip() for i in qrs[start+1:i]])
                    num=int(re.findall('\d\d*',item)[0])
                else:
                    num=int(re.findall('\d\d*',item)[0])
                    
            elif len(re.findall('.W',item)):
                start=i
        #-----------------------------------------------------------
        #relevance assessment
        temp=open(main_path+'/'+'cranqrel').readlines()
        temp=[item.split() for item in temp]
        qrs_keys=list(queries.keys())
        rel_ass={}
        rel_ranks={}
        rel_docs=[]
        rank=[]
        j=1
        for i in tqdm(temp):
            if int(i[0])!=j:
                rel_ass[qrs_keys[j-1]]=rel_docs
                rel_ranks[qrs_keys[j-1]]=rank
                rel_docs=[]
                rank=[]
                j+=1
            rel_docs.append(i[1])
            rank.append(i[2])
    #-----------------------------------------------------------
        #Documents
        docs={}
        filename = main_path+'cran.all.1400' 
        data=open(filename).read().split('.I ')
        data=[i for i in data if len(i)>0]
        for item in data:
            splitted=[i.strip().replace('\s\s*',' ').replace(' .','.').replace(' ,',',') for i in item.split()]
            num=int(splitted[0])
            title=splitted.index('.T')
            author=splitted.index('.A')
            book=splitted.index('.B')
            text=splitted.index('.W')
            if len(splitted[text+1:])>0:
                docs[num]=(' '.join(splitted[text+1:])).strip()
        return docs,queries, rel_ass