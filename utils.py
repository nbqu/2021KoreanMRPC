from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
import torch
import pandas as pd
import numpy as np
import random
import os

class MRPCDataset(Dataset):
    def __init__(self, PATH, pretrained_model):
        self.df = pd.read_csv(PATH,  delimiter='\t')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def __len__(self):
        return len(self.df)

    def add_sent_tokens(self, sentence):
        return self.tokenizer.bos_token + sentence + self.tokenizer.eos_token


class CoLADataset(MRPCDataset):
    def __init__(self, PATH, pretrained_model):
        super().__init__(PATH, pretrained_model)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        sentence, label = row.sentence, row.acceptability_label
        sentence = self.add_sent_tokens(sentence)
        sent_ids = self.tokenizer(sentence, padding='max_length', max_length=48) 
        sent_ids.data['input_ids'] = torch.as_tensor(sent_ids.data['input_ids'])
        sent_ids.data['attention_mask'] = torch.as_tensor(sent_ids.data['attention_mask'])
        label_tensor = torch.as_tensor(label)
        return sent_ids.data, label_tensor


class WiCDataset(MRPCDataset):
    def __init__(self, PATH, pretrained_model):
        super().__init__(PATH, pretrained_model)
        self.target_words = self.df.Target.tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, additional_special_tokens=self.target_words)
    
    def __getitem__(self, item):
        row = self.df.iloc[item]
        sent1, sent2, label = row.SENTENCE1, row.SENTENCE2, row.ANSWER
        sent1, sent2 = self.add_sent_tokens(sent1), self.add_sent_tokens(sent2)
        sent_ids = self.tokenizer(sent1, sent2, padding='max_length', max_length=384) # max length of train sentence = 142
        sent_ids.data['input_ids'] = torch.as_tensor(sent_ids.data['input_ids'])
        sent_ids.data['attention_mask'] = torch.as_tensor(sent_ids.data['attention_mask'])
        sent_ids.data['token_type_ids'] = torch.as_tensor([0]*len(sent1)+[1]*len(sent2))
        label_tensor = torch.as_tensor(label)

        return sent_ids.data, label_tensor


def file_selector(args):
    file_prefix = {
        'CoLA': 'NIKL_CoLA_',
        'WiC': 'NIKL_SKT_WiC_',
        'BoolQ': 'SKT_BoolQ_',
        'COPA': 'SKT_COPA_'
    }

    if args.task == 'CoLA':
        train_file = os.path.join(args.data_dir, file_prefix[args.task]+'train.tsv')
        valid_file = os.path.join(args.data_dir, file_prefix[args.task]+'dev.tsv')
        test_file = os.path.join(args.data_dir, file_prefix[args.task]+'test.tsv')
    else:
        train_file = os.path.join(args.data_dir, file_prefix[args.task]+'Train.tsv')
        valid_file = os.path.join(args.data_dir, file_prefix[args.task]+'Dev.tsv')
        test_file = os.path.join(args.data_dir, file_prefix[args.task]+'Test.tsv')
    
    if args.mode == 'train':
        return train_file, valid_file, None
    if args.mode == 'dev':
        return None, valid_file, None
    if args.mode == 'test':
        return None, None, test_file

def dataset_selector(file, args):
    task = args.task
    pretrained = args.pretrained_model
    if task == 'CoLA':
        return CoLADataset(file, pretrained)
    if task == 'WiC':
        return WiCDataset(file, pretrained)
    # if task == 'BoolQ':
    #     return BoolQDataset(file, pretrained)
    # if task == 'COPA':
    #     return COPADataset(file, pretrained)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

