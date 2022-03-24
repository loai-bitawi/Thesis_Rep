# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:24:11 2022

@author: Loai
"""

import json

import ijson

data = ijson.parse(open('E:/GitHub/Thesis/Lisa/docs.txt', 'r'))
keys=[]
for prefix, event, value in data:
    if event=='map_key':
       keys.append(str(value))
keys={'dict':keys}
with open('E:/GitHub/Thesis/Lisa/d_list.txt','w') as f:
    f.write(json.dumps(keys))


sample=list(json.loads(open('E:/GitHub/Thesis/Lisa/Sample_docs_q4_27smpl.txt').read()).keys())
sample=json.loads(open('E:/GitHub/Thesis/Lisa/sample1042.txt').read()).keys()
# sample=sample.replace(',','').replace('[','').replace(']','').replace(' ','')
# sample=[i for i in sample.split('"') if len(i)>0]
prfx={}
level1=''
level2=''
level3=''
level4=''
level5=''
l1=[]
l2=[]
l3=[]
l4=[]
l5=[]
i=0
temp=[]
temp2=[]
key=''
for prefix, event, value in data:

    if event=='map_key'and str(value) in sample:
        print(value)
        if len(prfx.keys()):
            prfx[key].append(l4)
            l3=[]
            l4=[]
        key=str(value)
        prfx[key]=[]
        level1=prefix
        level2=prefix+'.'+value
        level3=prefix+'.'+value+'.item'
        level4=prefix+'.'+value+'.item'+'.item'
        level5=prefix+'.'+value+'.item'+'.item'+'.item'
        # i+=1
        # if i ==4:break
    elif event=='map_key' and str(value) not in sample:
        continue
    if key in sample:    
        if prefix==level3:
            if event=='end_array':
                if len(temp):
                    l3.append(temp)
                    prfx[key]=l3
            if event=='start_array':
                temp=[]
    
        if prefix==level4:
            if event=='end_array':
                l4.append(temp2)
            if event=='start_array':
                temp2=[]
        if prefix==level4 and event=='string':
            temp.append(value)
        if prefix==level5 and event=='number':
            temp2.append(float(value))
    if event =='map_key' and value not in sample and len(prfx.keys())==len(sample):
        prfx[key].append(l4)
        break

data={}
data['dict']=prfx
with open('sample27.txt','w') as f:
    f.write(json.dumps(data))