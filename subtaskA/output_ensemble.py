import csv
import torch
from fairseq.models.roberta import RobertaModel
import sys
sys.path.append('/home/cqx/rubi/')
import aaa

with open('../data/Test Data/subtaskA_test_data.csv') as fsource, open('../subtaskA_generated/trial_hypo0.csv','w') as ffout:
    source=csv.reader(fsource)
    fout=csv.writer(ffout)
    next(source)
    ids=[]
    for idx,sent0,sent1 in source:
        ids.append([idx])
    fout.writerows(ids)

    fsource.close()
    ffout.close()


for j in range(9):
    print(j+1)
    roberta = RobertaModel.from_pretrained('../checkpoints/taskA/ensemble/%d'%(j+1), 'checkpoint_best.pt', '../../../../data')
    #roberta = RobertaModel.from_pretrained('..', 'checkpoints/taskA/ensemble/%d/checkpoint_best.pt'%(j+1), 'data')
    roberta.eval()  # disable dropout
    roberta.cuda()  # use the GPU (optional)
    
    with open('../data/Test Data/subtaskA_test_data.csv') as fsource, open('../subtaskA_generated/trial_hypo%d.csv'%(j+1),'w',newline='') as ffout,\
        open('../subtaskA_generated/trial_hypo%d.csv'%j) as fherit:
        source=csv.reader(fsource)
        herit=csv.reader(fherit)
        fout=csv.writer(ffout)
        next(source)
            
        for idx,sent0,sent1 in source:
            line1=next(herit)
            assert line1[0]==idx
            if sent0.isupper():
                sent0 = sent0.capitalize()
            if sent1.isupper():
                sent1 = sent1.capitalize()
            sent0 = roberta.encode(sent0)
            sent1 = roberta.encode(sent1)
            score0 = roberta.predict('sentence_classification_head', sent0, return_logits=True)
            score1 = roberta.predict('sentence_classification_head', sent1, return_logits=True)
            pred = torch.cat([score0,score1]).argmin().item()
            line1.append(pred)
            fout.writerow(line1)

        fsource.close()
        ffout.close()
        fherit.close()