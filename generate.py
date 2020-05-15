import torch
from fairseq.models.bart import BARTModel
import csv

bart = BARTModel.from_pretrained(
    'checkpoints/new',
    checkpoint_file='checkpoint4.pt',
    data_name_or_path='preprocess_taskC_data/subtaskC_data-bin'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open('data/Test Data/subtaskC_test_data_new_plusplus.csv') as source, open('subtaskC_generated/subtaskC_answers.csv', 'w') as fout:
#with open('data/Dev Data/subtaskC_dev_data_new_plusplus.csv') as source, open('subtaskC_generated/trial_hypo.csv', 'w') as fout:
#with open('data/Trial Data/taskC_trial_data.csv') as source, open('subtaskC_generated/trial_hypo.csv', 'w') as fout:
    source=csv.reader(source)
    fout=csv.writer(fout)
    next(source)
    idx,false_sent,true_sent,evidence_from_wik = next(source)
    #idx,false_sent = next(source)
    if false_sent.isupper():
        false_sent=false_sent.capitalize()
    if true_sent.isupper():
        true_sent=true_sent.capitalize()
    sline=false_sent
    #sline='The statement "'+false_sent+'" is absurd, because:'
    #sline='Context: '+evidence_from_wik+' | '+'The statement: '+false_sent
    #sline='Context: '+evidence_from_wik+' | '+'The statement "'+false_sent+'" is absurd, because:' #should be modified
    #sline='Context: '+evidence_from_wik+'\ Reasonable statement: '+ true_sent+' | '+'The statement "'+false_sent+'" is absurd, because:' #should be modified
    print(sline)
    indices = [idx]
    slines = [sline]
    for idx,false_sent,true_sent,evidence_from_wik in source:
    #for idx,false_sent in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, no_repeat_ngram_size=3)

            fout.writerows(zip(indices,hypotheses_batch))
            indices = []
            slines = []
        if false_sent.isupper():
            false_sent=false_sent.capitalize()
        if true_sent.isupper():
            true_sent=true_sent.capitalize()
        sline=false_sent
        #sline='The statement "'+false_sent+'" is absurd, because:'
        #sline='Context: '+evidence_from_wik+' | '+'The statement: '+false_sent
        #sline='Context: '+evidence_from_wik+' | '+'The statement "'+false_sent+'" is absurd, because:' #should be modified
        #sline='Context: '+evidence_from_wik+'\ Reasonable statement: '+ true_sent+' | '+'The statement "'+false_sent+'" is absurd, because:' #should be modified
        indices.append(idx)
        slines.append(sline)
        count += 1
    if slines != []:
        #hypotheses_batch = bart.sample(slines)
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, no_repeat_ngram_size=3)
        fout.writerows(zip(indices,hypotheses_batch))

#only false sentence : 3e-5 max tokens 1024(bz500+) best(epoch 09) bleu 18.9894     1e-5 bleu 18.6608 loss 4.85
#                                           last(epoch 19) bleu 18.2685
#                           max tokens 128(bz 69)  best(epoch 03) bleu 19.4463
#                           max tokens 128(bz 275) update 4 best bleu ...bz 275
#add true sentence : 3e-5 max token 256 (bz 51) bleu 18.9848 loss 4.68
#add wik sentence : 3e-5 max token 1024 (bz 32) bleu 20.0536 loss 4.67
# one third bleu 18.9857

#15.1522
#with multiTarget no extraWords 19.6646      test:19.4277
#with multiTarget no extraWords +evd 18.98
#with multiTarget + extraWords 18.0958
#+mT +evd  +extraWords 19.5036
#+mT +reasonable +extraWords 18.9848
#+mT +evd +reasonable +extraWords 20.0318   test:20.3853