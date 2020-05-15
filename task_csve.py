# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import csv
from itertools import islice

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
)
from fairseq.tasks import FairseqTask, register_task


@register_task('commonsense_ve')
class CommonsenseVETask(FairseqTask):
    """Task to finetune RoBERTa for Commonsense VE."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR',
                            help='path to data directory; we load csv')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--num-classes', type=int, default=2)

    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab
        self.mask = vocab.add_symbol('<mask>')

        self.bpe = encoders.build_bpe(args)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.criterion == 'sentence_ranking', 'Must set --criterion=sentence_ranking'

        # load data and label dictionaries
        vocab = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(vocab)))

        return cls(args, vocab)

    def load_dataset(self, split, epoch=0, combine=False, data_path=None, return_only=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def binarize(s, append_bos=False):
            if self.bpe is not None:
                s = self.bpe.encode(s)
            tokens = self.vocab.encode_line(
                s, append_eos=True, add_if_not_exist=False,
            ).long()
            if append_bos and self.args.init_token is not None:
                tokens = torch.cat([tokens.new([self.args.init_token]), tokens])
            return tokens

        # self.data_path_table={'train_input':os.path.join(self.args.data,'Training  Data','subtaskA_data_all.csv'),\
        #                       'train_answer':os.path.join(self.args.data,'Training  Data','subtaskA_answers_all.csv'),\
        #                       'valid_input':os.path.join(self.args.data,'Trial Data','taskA_trial_data.csv'),\
        #                       'valid_answer':os.path.join(self.args.data,'Trial Data','taskA_trial_answer.csv')\
        #                             }
        # self.data_path_table={'train_input':os.path.join(self.args.data,'trainval','subtaskA_data_all.csv'),\
        #                       'train_answer':os.path.join(self.args.data,'trainval','subtaskA_answers_all.csv'),\
        #                       'valid_input':os.path.join(self.args.data,'Dev Data','subtaskA_dev_data.csv'),\
        #                       'valid_answer':os.path.join(self.args.data,'Dev Data','subtaskA_gold_answers.csv')\
        #                             }
        self.data_path_table={'train_input':os.path.join(self.args.data,'trainvaldev','subtaskA_data_all_plusplus.csv'),\
                              'train_answer':os.path.join(self.args.data,'trainvaldev','subtaskA_answers_all.csv'),\
                              'valid_input':os.path.join(self.args.data,'Dev Data','subtaskA_dev_data_plusplus.csv'),\
                              'valid_answer':os.path.join(self.args.data,'Dev Data','subtaskA_gold_answers.csv')\
                                    }
        # self.data_path_table={'train_input':os.path.join(self.args.data,'subtaskA_data_all.csv'),\
        #                       'train_answer':os.path.join(self.args.data,'subtaskA_answers_all.csv'),\
        #                       'valid_input':os.path.join(self.args.data,'taskA_trial_data.csv'),\
        #                       'valid_answer':os.path.join(self.args.data,'taskA_trial_answer.csv')\
        #                             }
        data_path_input=self.data_path_table[split+'_input']
        data_path_answer=self.data_path_table[split+'_answer']
        
        if not os.path.exists(data_path_input):
            raise FileNotFoundError('Cannot find data: {}'.format(data_path_input))
        if not os.path.exists(data_path_answer):
            raise FileNotFoundError('Cannot find data: {}'.format(data_path_answer))

        src_tokens = [[] for i in range(self.args.num_classes)]
        src_lengths = [[] for i in range(self.args.num_classes)]
        src_ids = []
        labels = []
        label_ids = []

        with open(data_path_input) as f:
            reader=csv.reader(f)
            for row in islice(reader,1,None):
                src_ids.append(row[0])
                for i in range(self.args.num_classes):
                    src = row[i+1]
                    evidence = row[i+3]
                    if src.isupper():
                        src = src.capitalize()
                    src=src+' Context: '+evidence
                    src_bin = binarize(src,append_bos=True)
                    src_tokens[i].append(src_bin)
                    src_lengths[i].append(len(src_bin))

            assert all(len(src_tokens[0]) == len(src_tokens[i]) for i in range(self.args.num_classes))
            assert len(src_tokens[0]) == len(src_lengths[0])
       
        with open(data_path_answer) as f:
            reader=csv.reader(f)
            for row in reader:
                label_ids.append(row[0])
                label=1-int(row[1])
                labels.append(label)

            assert len(labels) == 0 or len(labels) == len(src_tokens[0])
            assert all(src_ids[i] == label_ids[i] for i in range(len(src_ids)))

        for i in range(self.args.num_classes):
            src_lengths[i] = np.array(src_lengths[i])
            src_tokens[i] = ListDataset(src_tokens[i], src_lengths[i])
            src_lengths[i] = ListDataset(src_lengths[i])

        dataset = {
            'id': IdDataset(),
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens[0], reduce=True),
        }

        for i in range(self.args.num_classes):
            dataset.update({
                'net_input{}'.format(i + 1): {
                    'src_tokens': RightPadDataset(
                        src_tokens[i],
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    'src_lengths': src_lengths[i],
                }
            })

        if len(labels) > 0:
            dataset.update({'target': RawLabelDataset(labels)})

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[np.maximum.reduce([src_token.sizes for src_token in src_tokens])],
        )

        with data_utils.numpy_seed(self.args.seed):
            dataset = SortDataset(
                dataset,
                # shuffle
                sort_order=[np.random.permutation(len(dataset))],
            )

        print('| Loaded {} with {} samples'.format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_classification_head(
            'sentence_classification_head',
            num_classes=1,
        )

        return model

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab
