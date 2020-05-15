import csv

# resource_path='../data/trainval/subtaskC_data_all_plusplus.csv'
# data_path='../data/trainval/subtaskB_data_all.csv'
# target_path='../data/trainval/subtaskB_data_all_plusplus.csv'

# resource_path='../data/Dev Data/subtaskC_dev_data_plusplus.csv'
# data_path='../data/Dev Data/subtaskB_dev_data.csv'
# target_path='../data/Dev Data/subtaskB_dev_data_plusplus.csv'

resource_path='../data/Test Data/subtaskC_test_data_plusplus.csv'
data_path='../data/Test Data/subtaskB_test_data.csv'
target_path='../data/Test Data/subtaskB_test_data_plusplus.csv'

with open(resource_path) as f1, open(data_path) as f2, open(target_path,'w',newline='') as f3:
    resource=csv.reader(f1)
    data=csv.reader(f2)
    target=csv.writer(f3)

    next(resource)
    header=list(next(data))
    target.writerow(header+['trueSent','wiktionary','urbandictionary'])
    for idx,FalseSent,trueSent,wiktionary,urbandictionary in resource:
        idx2,FalseSent2,OptionA,OptionB,OptionC=next(data)
        if not (idx==idx2 and FalseSent==FalseSent2):
            print(idx)
            print(FalseSent)
            print(FalseSent2)
            print(OptionA)
            print(OptionB)
            print(OptionC)
            target.writerow([idx2,FalseSent,OptionA,OptionB,OptionC,trueSent,wiktionary,urbandictionary])
        else:
            target.writerow([idx2,FalseSent2,OptionA,OptionB,OptionC,trueSent,wiktionary,urbandictionary])

f1.close()
f2.close()
f3.close()

