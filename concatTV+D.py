# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 01:19:01 2020

@author: WorldEditor
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:53:26 2020

@author: WorldEditor
"""


import csv

train_data_path='data/trainval/subtaskC_data_all_new_plusplus.csv'
valid_data_path='data/Dev Data/subtaskC_dev_data_new_plusplus.csv'
#dev_data_path='data/Dev Data/subtaskA_dev_data.csv'
#target_path='data/trainvaldev3/subtaskB_data_all_plusplus.csv'
target_path='data/trainvaldev3/subtaskC_data_all_new_plusplus.csv'

train_answer_path='data/trainval/subtaskC_answers_all.csv'
valid_answer_path='data/Dev Data/subtaskC_gold_answers.csv'
#dev_answer_path='data/Dev Data/subtaskA_gold_answers.csv'
#answer_target_path='data/trainvaldev3/subtaskB_answers_all.csv'
answer_target_path='data/trainvaldev3/subtaskC_answers_all.csv'

with open(train_data_path,encoding='utf-8') as f1, open(valid_data_path,encoding='utf-8') as f2,\
     open(target_path,'w',encoding='utf-8',newline='') as f3:
    train=csv.reader(f1)
    val=csv.reader(f2)
    target=csv.writer(f3)
    
    head=next(train)
    target.writerow(head)
    for line in train:
        target.writerow(line)
        
    next(val)
    for line in val:
        target.writerow(line)
        target.writerow(line)
        target.writerow(line)
        
    
        
f1.close()
f2.close()
f3.close()

with open(train_answer_path,encoding='utf-8') as f1, open(valid_answer_path,encoding='utf-8') as f2,\
     open(answer_target_path,'w',encoding='utf-8',newline='') as f3:
    train=csv.reader(f1)
    val=csv.reader(f2)
    target=csv.writer(f3)
    
    for line in train:
        target.writerow(line)
        
    for line in val:
        target.writerow(line)
        target.writerow(line)
        target.writerow(line)
        
        
        
f1.close()
f2.close()
f3.close()
    
    