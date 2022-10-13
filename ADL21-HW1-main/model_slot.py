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
                            bidirectional=self.bidirectional, batch_first=True, dropout=dropout) #除了self.embed之外,其他param的self都拔掉
        #self.fc = nn.Linear(hidden_size * 2, num_class)
        
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, 1024)
        # Fully-connected layer，把 hidden state 線性轉換成 output
        self.hidden2out = nn.Linear(self.hidden_size*2, self.num_classes) 

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn

        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        #x, _ = batch
        seq_len = batch.size()[1]
        out = self.embed(batch)  # [batch_size, seq_len, embeding]=[128, 32, 300],t()是轉置 (之後去掉)    
        H, _ = self.lstm(out)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha 
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        hiddenout = nn.Linear(self.hidden_size*2, seq_len*9) #9*seq_len,seq_len為一句的長度,9為類別數
        out = hiddenout(out)
        out = out.reshape(-1,128,9) #[seq_len,128,9]

        

        return out
        
