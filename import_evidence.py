import csv
import os

# data_path_table={'train_input':os.path.join('data','trainval','subtaskA_data_all_plusplus.csv'),\
#                 'train_input2':os.path.join('data','trainval','subtaskB_data_all_plusplus.csv'),\
#                'train_output':os.path.join('data','trainval','subtaskB_data_all_new_plusplus.csv'),\
#                'valid_input':os.path.join('data','Dev Data','subtaskA_dev_data_plusplus.csv'),\
#                 'valid_input2':os.path.join('data','Dev Data','subtaskB_dev_data_plusplus.csv'),\
#                'valid_output':os.path.join('data','Dev Data','subtaskB_dev_data_new_plusplus.csv')}

# data_path_table={'train_input':os.path.join('data','trainval','subtaskA_data_all_plusplus.csv'),\
#                 'train_input2':os.path.join('data','trainval','subtaskC_data_all_plusplus.csv'),\
#                'train_output':os.path.join('data','trainval','subtaskC_data_all_new_plusplus.csv'),\
#                'valid_input':os.path.join('data','Dev Data','subtaskA_dev_data_plusplus.csv'),\
#                 'valid_input2':os.path.join('data','Dev Data','subtaskC_dev_data_plusplus.csv'),\
#                'valid_output':os.path.join('data','Dev Data','subtaskC_dev_data_new_plusplus.csv')}

data_path_table={'test_input':os.path.join('data','Test Data','subtaskA_test_data_plusplus.csv'),\
                'test_input2':os.path.join('data','Test Data','subtaskC_test_data_plusplus.csv'),\
               'test_output':os.path.join('data','Test Data','subtaskC_test_data_new_plusplus.csv')}

#for split in ['train','valid']:
split='test'
input_path=data_path_table[split+'_input']
input_path2=data_path_table[split+'_input2']
output_path=data_path_table[split+'_output']
with open(input_path) as f1, open(input_path2) as f2, open(output_path,'w',newline='') as f3:
    reader=csv.reader(f1)
    reader2=csv.reader(f2)
    writer=csv.writer(f3)
    head=next(reader2)
    next(reader)
    writer.writerow(head[:-1])
    #for idx,FalseSent,OptionA,OptionB,OptionC,trueSent,wiktionary,urbandictionary in reader2:
    for idx,FalseSent,trueSent,wiktionary,urbandictionary in reader2:
        idx2,sent0,sent1,evidence_for_sent0,evidence_for_sent1=next(reader)
        assert idx==idx2
        if sent0==FalseSent:
            wiktionary=evidence_for_sent0
        elif sent1==FalseSent:
            wiktionary=evidence_for_sent1
        else:
            print(idx,FalseSent)
            print("sent0:",sent0)
            print("sent1:",sent1)
            a=int(input("which one?"))
            if a == 0:
                wiktionary=evidence_for_sent0
            elif a==1:
                wiktionary=evidence_for_sent1
            else:
                raise RuntimeError("invalid number!!!")
        #writer.writerow([idx,FalseSent,OptionA,OptionB,OptionC,trueSent,wiktionary])
        writer.writerow([idx,FalseSent,trueSent,wiktionary])
f1.close()
f2.close()
f3.close()