from asyncio.windows_events import NULL
from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

import torch



class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[List[str], int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping #本身就是label2idx
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance


    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)
    
    

    

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        label_list, text_list, id_list = [], [], []
        for sample in samples:
            label = [ self.label_mapping[tag] for tag in sample['tags'] ] #1個個轉換
            label_list.append(label)
            id_list.append(sample['id'])
            text_list.append(sample['tokens']) #List[List[str]]
        
        text =  self.vocab.encode_batch(text_list) 
        processed_text = torch.tensor(text, dtype=torch.int64) 
        
        
        label_list = padding(label_list)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        

        collate ={
            "label":label_list,
            "input":processed_text,
            "id":id_list
        }
        
        return collate 
        

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

    
def padding(batch: List[List[int]]):
    to_len = max(len(tag) for tag in batch)
    paddeds = [seq[:to_len] + [0] * max(0, to_len - len(seq)) for seq in batch]
    return paddeds