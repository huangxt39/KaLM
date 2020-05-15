import csv
import torch
from fairseq.models.roberta import RobertaModel
import sys
sys.path.append('/home/cqx/rubi/')
import aaa
roberta = RobertaModel.from_pretrained('../checkpoints/falseOnlySmallerBZ', 'checkpoint_best.pt', '../../data')
roberta.eval()  # disable dropout
roberta.cuda()  # use the GPU (optional)
with open('../data/Test Data/subtaskB_test_data_plusplus_m.csv') as source, open('../subtaskB_generated/trial_hypo.csv','w') as fout:
    table=['A','B','C']
    source=csv.reader(source)
    fout=csv.writer(fout)
    next(source)
    indices=[]
    hypotheses=[]
    for idx,FalseSent,OptionA,OptionB,OptionC,trueSent,wiktionary,urbandictionary in source:
        sents=[]
        scores=[]
        indices.append(idx)
        if FalseSent.isupper():
                FalseSent = FalseSent.capitalize()
        FalseSent = 'The statement "'+ FalseSent + '" is absurd.'
        for option in (OptionA,OptionB,OptionC):
            if option.isupper():
                option = option.capitalize()
            sents.append(FalseSent+' Because '+option)

        for sent in sents:
            sent = roberta.encode(sent)
            score = roberta.predict('sentence_classification_head', sent, return_logits=True)
            scores.append(score)
        pred = torch.cat(scores).argmax().item()
        hypotheses.append(table[pred])

    fout.writerows(zip(indices,hypotheses))