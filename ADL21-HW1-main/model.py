from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn
import numpy as np



class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)     #      # 字向量维度
        #設定
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_class
        self.bidirectional = bidirectional
        
        self.embedding_dim = 300 #每個詞都用300維來表示
            
        

        # TODO: model architecture
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, batch_first=True, dropout=dropout) #除了self.embed之外,其他param的self都拔掉
        #self.fc = nn.Linear(hidden_size * 2, num_class)
        # Fully-connected layer，把 hidden state 線性轉換成 output
        self.hidden2out = nn.Linear(self.hidden_size*2, self.num_classes) #這個應該是最後一層,輸出每一種可能的機率

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn

        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        #x, _ = batch
        out = self.embed(batch.t())  # [batch_size, seq_len, embeding]=[128, 32, 300],t()是轉置
        out, _ = self.lstm(out)
        ht = out[-1]  # 句子最後时刻的 hidden state  out[:, -1, :]
        #out = nn.functional.softmax(out,self.num_classes)
        out = self.hidden2out(ht)
        

        return out
        
