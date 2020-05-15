import csv

with open('../subtaskA_generated/trial_hypo9.csv') as fresult, open('../subtaskA_generated/final_result.csv','w',newline='') as ffinal:
    result=csv.reader(fresult)
    final=csv.writer(ffinal)

    for line in result:
        idx=line[0]
        line2=[idx]
        ones=0
        zeros=0
        for item in line[1:]:
            if item=='1':
                ones+=1
            if item=='0':
                zeros+=1
        assert ones+zeros==9
        if ones>zeros:
            line2.append('1')
        else:
            line2.append('0')
        final.writerow(line2)

    fresult.close()
    ffinal.close()


