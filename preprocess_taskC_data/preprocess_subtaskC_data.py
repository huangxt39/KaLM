import csv
from itertools import islice
import os

data_path=os.path.join('../data','trainvaldev3','subtaskC_data_all_new_plusplus.csv')
#data_path=os.path.join('../data','trainval','subtaskC_data_all_new_plusplus.csv')
data_dest_path=os.path.join('subtaskC_data','train.source')
data=[]
with open(data_path) as f:
    reader=csv.reader(f)
    for idx,false_sent,true_sent,evidence_from_wik in islice(reader,1,None):
    #for idx,false_sent in islice(reader,1,None):
        if false_sent.isupper():
            false_sent=false_sent.capitalize()
        if true_sent.isupper():
            true_sent=true_sent.capitalize()
        sequence=false_sent
        #sequence='The statement "'+false_sent+'" is absurd, because:' 
        #sequence='Context: '+evidence_from_wik+' | '+'The statement: '+false_sent
        #sequence='Context: '+evidence_from_wik+' | '+'The statement "'+false_sent+'" is absurd, because:' #should be modified
        #sequence='Context: '+evidence_from_wik+'\ Reasonable statement: '+ true_sent+' | '+'The statement "'+false_sent+'" is absurd, because:' #should be modified
        data.append(sequence)

with open(data_dest_path,'w') as f:
    for sentence in data:
        f.write(sentence+'\n')
        f.write(sentence+'\n')
        f.write(sentence+'\n')

len1=len(data)*3
f.close()

data_path=os.path.join('../data','trainvaldev3','subtaskC_answers_all.csv')
#data_path=os.path.join('../data','trainval','subtaskC_answers_all.csv')
data_dest_path=os.path.join('subtaskC_data','train.target')
data=[]
with open(data_path) as f:
    reader=csv.reader(f)
    for idx,sent0,sent1,sent2 in reader:
        if sent0.isupper():
            sent0=sent0.capitalize()
        if sent1.isupper():
            sent1=sent1.capitalize()
        if sent2.isupper():
            sent2=sent2.capitalize()
        data.append(sent0)
        data.append(sent1)
        data.append(sent2)

with open(data_dest_path,'w') as f:
    for sentence in data:
        f.write(sentence+'\n')

len2=len(data)
f.close()
assert len1==len2

data_path=os.path.join('../data','Dev Data','subtaskC_dev_data_new_plusplus.csv')
#data_path=os.path.join('../data','Trial Data','taskC_trial_data.csv')
data_dest_path=os.path.join('subtaskC_data','val.source')
data=[]
with open(data_path) as f:
    reader=csv.reader(f)
    for idx,false_sent,true_sent,evidence_from_wik in islice(reader,1,None):
    #for idx,false_sent in islice(reader,1,None):
        if false_sent.isupper():
            false_sent=false_sent.capitalize()
        if true_sent.isupper():
            true_sent=true_sent.capitalize()
        sequence=false_sent
        #sequence='The statement "'+false_sent+'" is absurd, because:' #should be modified
        #sequence='Context: '+evidence_from_wik+' | '+'The statement: '+false_sent #should be modified
        #sequence='Context: '+evidence_from_wik+' | '+'The statement "'+false_sent+'" is absurd, because:' #should be modified
        #sequence='Context: '+evidence_from_wik+'\ Reasonable statement: '+ true_sent+' | '+'The statement "'+false_sent+'" is absurd, because:' #should be modified
        data.append(sequence)

with open(data_dest_path,'w') as f:
    for sentence in data:
        f.write(sentence+'\n')
        f.write(sentence+'\n')
        f.write(sentence+'\n')

len1=len(data)*3
f.close()

data_path=os.path.join('../data','Dev Data','subtaskC_gold_answers.csv')
#data_path=os.path.join('../data','Trial Data','taskC_trial_references.csv')
data_dest_path=os.path.join('subtaskC_data','val.target')
data=[]
with open(data_path) as f:
    reader=csv.reader(f)
    for idx,sent0,sent1,sent2 in reader:
        if sent0.isupper():
            sent0=sent0.capitalize()
        if sent1.isupper():
            sent1=sent1.capitalize()
        if sent2.isupper():
            sent2=sent2.capitalize()
        data.append(sent0)
        data.append(sent1)
        data.append(sent2)

with open(data_dest_path,'w') as f:
    for sentence in data:
        f.write(sentence+'\n')

len2=len(data)
f.close()
assert len1==len2