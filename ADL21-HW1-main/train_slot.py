import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange
from torch.utils.data import DataLoader

from dataset_slot import SeqClsDataset
from utils import Vocab

from model_slot import SeqClassifier 
import torch.optim as optim
import os



TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]  #拆分成訓練及驗證data




def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text()) #所有的slot和其編號的dict

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS} #產生 訓練和驗證的資料路徑
    #print(data_paths)
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    #print(data)
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, slot2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    #dataloader = {DataLoader(dataset=datasets[split], batch_size=4, shuffle=True) for split in SPLITS}
    dataloader_train = DataLoader(dataset=datasets["train"], batch_size=args.batch_size ,shuffle=True,collate_fn=datasets["train"].collate_fn)
    dataloader_eval = DataLoader(dataset=datasets["eval"], batch_size= args.batch_size, shuffle=True,collate_fn=datasets["eval"].collate_fn)

    


    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqClassifier(embeddings,args.hidden_size,args.num_layers,args.dropout,args.bidirectional,9).to(device) 
    '''
    FILE3 = 'model_all_slot.pt'
    model = torch.load(FILE3) #要把第53行的model註解才行,這個能直接延用上一個model參數
    '''
    
    #train_iter = build_iterator(datasets["train"],batch_size=4, device=device)#迭代用,dataset先不用loader
    # TODO: init optimizer
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer =optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()


    epoch_pbar = trange(args.num_epoch, desc="Epoch") #進度條

    
    loss_record = []
    times = 0
    for epoch in epoch_pbar:
        #print(f"times ={times}")
        times = 0
        false_times = 0
        true_times = 0
        # TODO: Training loop - iterate over train dataloader and update model weights    
        #print(dataloader_train)
        for num,data in enumerate(dataloader_train):  #訓練集 train_iter
            times+=1
            #print("label = ",data['label'])
            #print("input = ",data['input'])
            input =  data['input'].to(device)
            label = data['label'].to(device)
            #print(input)
            #print(label)
            output = model(input)
            #print(output)
            #seq_len = label.size()[1]
            #label_batch_size = label.size()[0]#因為可能有不滿128的
            

            loss = loss_fn(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #_, preds = torch.max(output, 1)
            '''
            for i in range(128):
                if(label[i] != preds[i]):
                    false_times+=1
                else:
                    true_times+=1
                    '''
            preds = torch.max(output,1)[1].t() #[0]是max的機率,[1]是其index
            true_times  =  torch.sum(preds == label.data)
            false_times = len(label)-true_times  #正確率不是一句全對才對,而是只要各自單字有一個對,他自己就對
            loss_record.append(loss)
            print(loss) 
            
            print(f"第{epoch}週期 第{times}次的正確率 = {true_times}/{true_times+false_times} = {true_times/(true_times+false_times)}")
        #print(f"final正確率 = {true_times}/{true_times+false_times} = {true_times/(true_times+false_times)}")

    # TODO: Inference on test set 作測驗用
    print("before save")
    FILE = os.path.join(args.ckpt_dir,'model_state_dict_slot.pt')
    torch.save(model.state_dict(),FILE) #儲存模型,不知為何args.ckpt_dir不行,先生出之後在手動挪

    '''
    # save whole model (先存,這樣能夠反覆train)
    FILE = os.path.join(args.ckpt_dir,'model_all_slot.pt')
    torch.save(model, FILE)
    '''


    #驗證
    # TODO: Training loop - iterate over train dataloader and update model weights    
    #print(dataloader_train)
    total_true = 0
    total_false = 0
    for num,data in enumerate(dataloader_eval):  #訓練集 train_iter
        #print("label = ",data['label'])
        print("input = ",data['input'])
        input =  data['input'].to(device)
        label = data['label'].to(device)

        output = model(input)
        _, preds = torch.max(output, 1)
        true_times  =  torch.sum(preds == label.data)
        total_true += true_times
        false_times = len(label)-true_times
        total_false += false_times
        loss_record.append(loss)
        print(loss)    
        print(f"eval 正確率 = {true_times}/{true_times+false_times} = {true_times/(true_times+false_times)}")
    print(f"eval final正確率 = {total_true}/{total_true+total_false} = {total_true/(total_true+total_false)}")



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=15)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
