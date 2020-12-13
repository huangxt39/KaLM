# KaLM

This repo contains the code, data, and results for the [paper](https://www.aclweb.org/anthology/2020.semeval-1.67) "KaLM at SemEval-2020 Task 4: Knowledge-aware Language Models for Comprehension And Generation"

## Something Important
The official released results of SemEval-2020 Task 4 is [here](http://bit.ly/semeval2020-task4-results). Our team name is "LuoJunNB". (LuoJun is the name of the president of SYSU)

We provide the evidence that obtained by our searching approach. The evidence is in the files with a suffix "_new_plusplus.csv" in directory "./data" , e.g., "./data/Dev Data/subtaskC_dev_data_new_plusplus.csv".
  (the files with a suffix "_plusplus.csv" is obtained by older version of our searching approach used in the competition, while the files with suffix "_new_plusplus.csv" is obtained by the optimized version that we used in our experiments after the competition)

We put our best result (BLEU 20.39) for subtask C in './subtaskC_generated/subtaskC_answers.csv", which corresponds to the last row of table 4 in our paper.

## Introduction

"train.sh", "train2.sh", "train3.sh" are used to train the model for subtask A, B, and C respectively.

"task_csve.py" is the task file (a component to run model in Fairseq) for subtask A, which is mainly uesd to process data.

"task_csve2.py" is the task file for subtask B.

For subtask C, we do not write a customized task file. We preprocess the data in "./preprocess_taskC_data/"

"add_evidence.py" is the core file where we implemented our evidence searching approach.
