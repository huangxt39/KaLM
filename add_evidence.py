# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 03:13:23 2020

@author: WorldEditor
"""
import os
import csv
import requests
import spacy
import re

nlp=spacy.load('en_core_web_sm')
def find_evidence(sent): 
    sent=sent.replace('thirtieth','30th')
    #sent=sent.replace('30th','30')
    sent=sent.replace('thirty-first','31st')
    #sent=sent.replace('31st','31')
    doc=nlp(sent)
    key_words=[]
    chunks=[]
    for token in doc:
        if token.pos_=='VERB' and token.is_stop is not True:
            chunks.append(token.lemma_)
            
    for chunk in doc.noun_chunks:
        is_name=False
        chunk_text=nlp(chunk.text)
        for ent in chunk_text.ents:
            if ent.label_=='PERSON':
                is_name=True
                break
        if is_name:
            continue
        #key_words.append(chunk.root.text)
        
        tokens=[]
        for token in chunk_text:
            if token.pos_!='DET' and token.pos_!='PRON' and token.lemma_!='people' and token.lemma_!='People':
                tokens.append(token.text)
                #key_words.append(token.lemma_)
        if len(tokens):
            chunks.append(' '.join(tokens))
    
    for i in range(len(chunks)):
        key_words.append([])
    
    for chunk in chunks:
        words=chunk.split()
        for i,other_chunk in enumerate(chunks):
            if other_chunk !=chunk:
                key_words[i]=key_words[i]+words
    
    return chunks,key_words

def search_wiktionary(chunks,key_words_list,subsearch=False):
    header={'Content-Type':'application/json'}
    evidence_from_wik=''
    additional_n=0 if subsearch else max(3-len(chunks),0)
    for i,chunk in enumerate(chunks):
        #key_words=key_words_list[i]
        word=''
        for item in chunk.split():
            word+='{"match":{"word":{"query":"%s","boost":10}}},'%item
        #for key in key_words:
            #word+='{"match":{"gloss":{"query":"%s","boost":2}}},'%key
        word+='{"match":{"important":{"query":1,"boost":2}}}'
        json_body='''
                    {
                    "query":{
                        "bool":{
                            "should":[%s]
                            }
                        }
                    }'''%word
        try:
            response=requests.get("http://localhost:9200/wiktionary/_search",headers=header, data=json_body).json()
            result_list=response['hits']['hits']
        except:
            result_list=[]
        
        count=0
        i=0
        searched_subwords=[]
        while count<2+additional_n:
            if i==len(result_list):
                break
            word=result_list[i]['_source']['word']
            gloss=result_list[i]['_source']['gloss']
            i+=1
            if word is not None and gloss is not None:
                remove_g=re.search(r"(initialism|historical|obsolete|abbreviation|\(dated\)|slang|acronym|\(US\)|synonym|archaic|surname|\(rare\))",gloss)
                remove_w=re.search(r"-",word)
                remove_w2=re.match(r"[A-Z].+?",word)
                if remove_g is None and remove_w is None and remove_w2 is None:
                    count+=1
                    evidence_from_wik=evidence_from_wik+word+': '+gloss+' \\ '  #it would be nice to add a ['sep']
                    get_prototype=re.search(r'(plural of|past of|third person singular of|clipping of|alternative form of|alternative spelling of) "(.+)"',gloss)
                    # if get_prototype is not None:
                    #     print(get_prototype.group())
                    if get_prototype is not None and not subsearch and get_prototype.group(2) not in searched_subwords:
                        evidence_of_prototype=search_wiktionary([get_prototype.group(2)],None,subsearch=True)
                        evidence_from_wik+=evidence_of_prototype
                        searched_subwords.append(get_prototype.group(2))
                else:
                    print(word,gloss)
    return evidence_from_wik

def search_urbandictionary(chunks,key_words_list):
    header={'Content-Type':'application/json'}
    evidence_from_urb=''
    for i,chunk in enumerate(chunks):
        #key_words=key_words_list[i]
        word=''
        for item in chunk.split():
            word+='{"match":{"word":{"query":"%s","boost":10}}},'%item
        #for key in key_words:
            #word+='{"match":{"definition":{"query":"%s","boost":2}}},'%key
        json_body='''
                    {
                    "query":{
                        "function_score":{
                                "query":{
                                        "bool":{
                                            "should":[%s]
                                            }
                                        },
                                "field_value_factor": {
                                    "field":    "up_votes",
                                    "modifier": "log1p",
                                    "factor":   0.1
                                  },
                                  "boost_mode": "sum"
                            }        
                        }
                    }'''%word[:-1]
        response=requests.get("http://localhost:9200/knowledgebase/_search",headers=header, data=json_body).json()
        try:
            result_list=response['hits']['hits']
        except KeyError:
            result_list=[]
        for i in range(min(2,len(result_list))):
            word=result_list[i]['_source']['word']
            gloss=result_list[i]['_source']['definition']
            evidence_from_urb=evidence_from_urb+word+': '+gloss+' \\ '  #it would be nice to add a ['sep']
    return evidence_from_urb


# data_path_table={'train_input':os.path.join('data','trainval','subtaskC_data_all_plus.csv'),\
#                'train_output':os.path.join('data','trainval','subtaskC_data_all_plusplus.csv'),\
#                'valid_input':os.path.join('data','Dev Data','subtaskC_dev_data_plus.csv'),\
#                'valid_output':os.path.join('data','Dev Data','subtaskC_dev_data_plusplus.csv')}

# for split in ['train','valid']:
#     input_path=data_path_table[split+'_input']
#     output_path=data_path_table[split+'_output']
#     with open(input_path) as f1, open(output_path,'w',newline='') as f2:
#         reader=csv.reader(f1)
#         writer=csv.writer(f2)
#         head=next(reader)
#         writer.writerow(head+['wiktionary','urbandictionary'])
#         for idx,false_sent,true_sent, in reader:
#             chunks,key_words=find_evidence(false_sent)
#             evidence_from_wik=search_wiktionary(chunks,key_words)
#             lis=evidence_from_wik.split(' ')
#             lis=lis[:200] if len(lis)>200 else lis
#             evidence_from_wik=' '.join(lis)
#             evidence_from_urb=search_urbandictionary(chunks,key_words)
#             lis=evidence_from_urb.split(' ')
#             lis=lis[:200] if len(lis)>200 else lis
#             evidence_from_urb=' '.join(lis)
#             writer.writerow([idx,false_sent,true_sent,evidence_from_wik,evidence_from_urb])
#     f1.close()
#     f2.close()


# input_path=os.path.join('data','Test Data','subtaskC_test_data_plus.csv')
# output_path=os.path.join('data','Test Data','subtaskC_test_data_plusplus.csv')

# with open(input_path) as f1, open(output_path,'w',newline='') as f2:
#     reader=csv.reader(f1)
#     writer=csv.writer(f2)
#     head=next(reader)
#     writer.writerow(head+['wiktionary','urbandictionary'])
#     for idx,false_sent,true_sent, in reader:
#         chunks,key_words=find_evidence(false_sent)
#         evidence_from_wik=search_wiktionary(chunks,key_words)
#         lis=evidence_from_wik.split(' ')
#         lis=lis[:200] if len(lis)>200 else lis
#         evidence_from_wik=' '.join(lis)
#         evidence_from_urb=search_urbandictionary(chunks,key_words)
#         lis=evidence_from_urb.split(' ')
#         lis=lis[:200] if len(lis)>200 else lis
#         evidence_from_urb=' '.join(lis)
#         writer.writerow([idx,false_sent,true_sent,evidence_from_wik,evidence_from_urb])
# f1.close()
# f2.close()

        
# data_path_table={'train_input':os.path.join('data','trainval','subtaskA_data_all.csv'),\
#                'train_output':os.path.join('data','trainval','subtaskA_data_all_plusplus.csv'),\
#                'valid_input':os.path.join('data','Dev Data','subtaskA_dev_data.csv'),\
#                'valid_output':os.path.join('data','Dev Data','subtaskA_dev_data_plusplus.csv')}

# for split in ['train','valid']:
#     input_path=data_path_table[split+'_input']
#     output_path=data_path_table[split+'_output']
#     with open(input_path) as f1, open(output_path,'w',newline='') as f2:
#         reader=csv.reader(f1)
#         writer=csv.writer(f2)
#         head=next(reader)
#         writer.writerow(head+['wiktionary_sent0','wiktionary_sent1'])
#         for idx,sent0,sent1, in reader:
#             chunks,key_words=find_evidence(sent0)
#             evidence_for_sent0=search_wiktionary(chunks,key_words)
#             lis=evidence_for_sent0.split(' ')
#             lis=lis[:200] if len(lis)>200 else lis
#             evidence_for_sent0=' '.join(lis)
#             chunks,key_words=find_evidence(sent1)
#             evidence_for_sent1=search_wiktionary(chunks,key_words)
#             lis=evidence_for_sent1.split(' ')
#             lis=lis[:200] if len(lis)>200 else lis
#             evidence_for_sent1=' '.join(lis)
#             writer.writerow([idx,sent0,sent1,evidence_for_sent0,evidence_for_sent1])
#     f1.close()
#     f2.close()

input_path=os.path.join('data','Test Data','subtaskA_test_data.csv')
output_path=os.path.join('data','Test Data','subtaskA_test_data_plusplus.csv')

with open(input_path) as f1, open(output_path,'w',newline='') as f2:
    reader=csv.reader(f1)
    writer=csv.writer(f2)
    head=next(reader)
    writer.writerow(head+['wiktionary_sent0','wiktionary_sent1'])
    for idx,sent0,sent1, in reader:
            chunks,key_words=find_evidence(sent0)
            evidence_for_sent0=search_wiktionary(chunks,key_words)
            lis=evidence_for_sent0.split(' ')
            lis=lis[:200] if len(lis)>200 else lis
            evidence_for_sent0=' '.join(lis)
            chunks,key_words=find_evidence(sent1)
            evidence_for_sent1=search_wiktionary(chunks,key_words)
            lis=evidence_for_sent1.split(' ')
            lis=lis[:200] if len(lis)>200 else lis
            evidence_for_sent1=' '.join(lis)
            writer.writerow([idx,sent0,sent1,evidence_for_sent0,evidence_for_sent1])
f1.close()
f2.close()