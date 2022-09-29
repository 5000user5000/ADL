from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn



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
        self.embed = Embedding.from_pretrained(embeddings, freeze=False) 
        print(self.embed)
        # TODO: model architecture
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(3*4, 120)  #一個資料裡面有3項目*4 batch
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = nn.relu(self.fc1(batch))
        x = nn.relu(self.fc2(x))
        x = nn.Sigmoid(self.fc3(x))

        return x
        raise NotImplementedError
