import os
import json
import numpy as np
from itertools import chain
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
import datasets

class MultiChoiceDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, str],
        type: str,
        tokenizer: AutoTokenizer,
        max_len: int = 512,
        pad_to_max_len: bool = False,
    ):
    
        #type是指資料種類,如train  valid test
        self.type = type 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_max_len = pad_to_max_len

        self.context_path = data['context']
        #根据資料種類去拿相關data
        self.data_path = data[type] 
        #預處理後的資料
        self.data = self.preprocess()
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def preprocess(self) -> List[Dict]:
        #打開文件
        with open(self.context_path, 'r', encoding='utf-8') as context_f:
            raw_context: List = json.load(context_f)
        with open(self.data_path, 'r', encoding='utf-8') as data_f:
            raw_data: List[Dict] = json.load(data_f)

        #根據不同種類的資料,給予不同的返回值,這樣就不用寫2個dataset
        if self.type == 'test':
            ret = [{
                'question': [sample['question']] * len(sample['paragraphs']),
                'context': [raw_context[idx] for idx in sample['paragraphs']],
            } for sample in raw_data]
        else:
            ret = [{
                'question': [sample['question']] * len(sample['paragraphs']),
                'context': [raw_context[idx] for idx in sample['paragraphs']],
                'label': sample['paragraphs'].index(sample['relevant']),
            } for sample in raw_data]

        return ret


    def collate(self,_) -> Dict[str, torch.Tensor]:
        # 參數的 '_' 只是佔個位置,沒有用途,因為輸入參數預設為2個,但是實際只需要self就好 

        #預設2
        batch_size =  2 
        #預設4,也就是有4個選項
        num_choices = 4 
        
        # extract data
        question_set: List[List] = [instance['question'] for instance in self.data]
        context_set: List[List] = [instance['context'] for instance in self.data]
        if self.type != 'test':
            labels: List = [instance['label'] for instance in self.data]

        # flatten input
        questions = list(chain(*question_set))
        contexts = list(chain(*context_set))

        # tokenize
        batch = self.tokenizer(
            questions,
            contexts,
            truncation=True,
            max_length=self.max_len,
            padding="max_length" if self.pad_to_max_len else "longest",
            return_tensors="pt"
        )

        # un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # add lable
        if self.type != 'test':
            batch['labels'] = torch.tensor(labels)

        return batch
    
