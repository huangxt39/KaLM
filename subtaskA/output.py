import csv
import torch
from fairseq.models.roberta import RobertaModel
import sys
sys.path.append('/home/cqx/rubi/')
import aaa
#roberta = RobertaModel.from_pretrained('../checkpoints/taskA/captalize', 'checkpoint_best.pt', '../../../data')
roberta = RobertaModel.from_pretrained('../checkpoints/taskA/trainvaldev', 'checkpoint_last.pt', '../../../data')
roberta.eval()  # disable dropout
roberta.cuda()  # use the GPU (optional)
#with open('../data/Trial Data/taskA_trial_data.csv') as source, open('../subtaskA_generated/trial_hypo.csv','w') as fout:
with open('../data/Test Data/subtaskA_test_data_plusplus.csv') as source, open('../subtaskA_generated/subtaskA_answers.csv','w') as fout:
    source=csv.reader(source)
    fout=csv.writer(fout)
    next(source)
    indices=[]
    hypotheses=[]
    for idx,sent0,sent1,evd_sent0,evd_sent1 in source:
        indices.append(idx)
        if sent0.isupper():
            sent0 = sent0.capitalize()
        #sent0=sent0+' Context: '+evd_sent0
        if sent1.isupper():
            sent1 = sent1.capitalize()
        #sent1=sent1+' Context: '+evd_sent1
        sent0 = roberta.encode(sent0)
        sent1 = roberta.encode(sent1)
        score0 = roberta.predict('sentence_classification_head', sent0, return_logits=True)
        score1 = roberta.predict('sentence_classification_head', sent1, return_logits=True)
        pred = torch.cat([score0,score1]).argmin().item()
        hypotheses.append(pred)

    fout.writerows(zip(indices,hypotheses))