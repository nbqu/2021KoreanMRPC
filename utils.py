from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
import torch
import pandas as pd
import numpy as np
import random



class CoLADataset(Dataset):
    def __init__(self, PATH, pretrained_model):
        self.df = pd.read_csv(PATH,  delimiter='\t')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        sentence, label = row.sentence, row.acceptability_label
        sentence = self.tokenizer.bos_token + sentence + self.tokenizer.eos_token
        sent_ids = self.tokenizer(sentence, padding='max_length', max_length=48) 
        sent_ids.data['input_ids'] = torch.as_tensor(sent_ids.data['input_ids'])
        sent_ids.data['attention_mask'] = torch.as_tensor(sent_ids.data['attention_mask'])
        label_tensor = torch.as_tensor(label)
        return sent_ids.data, label_tensor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

