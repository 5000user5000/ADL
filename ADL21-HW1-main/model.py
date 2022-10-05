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
        
        self.embedding_pretrained = embeddings
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        #隨便設定
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.5
        self.num_classes = 150
        #print(self.embed)
        
        # TODO: model architecture
        self.lstm = nn.LSTM(self.embed, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout) #除了self.embed之外,其他param的self都拔掉
        self.fc = nn.Linear(hidden_size * 2, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最後时刻的 hidden state
        return out
        raise NotImplementedError
