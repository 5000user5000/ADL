from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn
import torch.nn.functional as F



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
                            bidirectional=self.bidirectional, batch_first=True) #除了self.embed之外,其他param的self都拔掉
        #self.fc = nn.Linear(hidden_size * 2, num_class)
        
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(hidden_size * 2, 9)
        # Fully-connected layer，把 hidden state 線性轉換成 output
        self.hidden2out = nn.Linear(self.hidden_size*2, self.num_classes) 

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn

        raise NotImplementedError

    def forward(self, X) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        #x, _ = batch
        #batch_size = batch.size()[0]
        #seq_len = batch.size()[1]
        out = self.embed(X)
        embed = F.dropout(out, p=0.2, training=True) 
        outputs, _ = self.lstm(embed)
        outputs = self.fc(outputs)
        

        return outputs
        
