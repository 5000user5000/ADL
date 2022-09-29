import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from utils import Vocab

from model import SeqClassifier # model.py
import torch.optim as optim




TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]  #拆分成訓練及驗證data




def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text()) #所有的intent和其編號的dict

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS} #產生 訓練和驗證的資料路徑
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    #dataloader = {DataLoader(dataset=datasets[split], batch_size=4, shuffle=True) for split in SPLITS}
    dataloader_train = DataLoader(dataset=datasets["train"], batch_size=4, shuffle=True)
    dataloader_eval = DataLoader(dataset=datasets["eval"], batch_size=4, shuffle=True)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqClassifier(embeddings,3,3,0.2,True,args.max_len).to(device) #param我先隨便設定

    # TODO: init optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch") #進度條

    


    for epoch in epoch_pbar:
        print(f"epoch1 = {epoch}")
        false_times = 0
        true_times = 0
        # TODO: Training loop - iterate over train dataloader and update model weights    
        for input,target in dataloader_train:  #訓練集
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            
            loss = loss_fn(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(input != output):
                false_times+=1
            else:
                true_times+=1
        print(f"正確率 = {true_times}/{true_times+false_times} = {true_times/(true_times+false_times)}")


    
    for epoch in epoch_pbar:
        print(f"epoch2 = {epoch}")
        # TODO: Evaluation loop - calculate accuracy and save model weights
        for input,target in dataloader_eval:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            #loss = loss_fn(output,target)
        
        if(input != output):
                false_times+=1
        else:
                true_times+=1
        print(f"正確率 = {true_times}/{true_times+false_times} = {true_times/(true_times+false_times)}")
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
