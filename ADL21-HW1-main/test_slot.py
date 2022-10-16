import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List
from torch.utils.data import DataLoader
import torch

from dataset_slot import SeqClsDataset
from model_slot import SeqClassifier
from utils import Vocab
import csv


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    idx2label = {idx: intent for intent, idx in intent2idx.items()}
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: create DataLoader for test dataset
    dataloader_test = DataLoader(dataset=dataset, batch_size=args.batch_size ,shuffle=False,collate_fn=dataset.collate_fn_test)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path,map_location='cpu')
    # load weights into model
    model.load_state_dict(ckpt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # TODO: predict dataset
    result = []
    for num,data in enumerate(dataloader_test):  
            input =  data['input'].to(device)
            output = model(input)
            _, preds = torch.max(output, 1)
            #print(len(preds))
            _pred = []
            for i in range(len(preds)):
                _pred.append(idx2label[int(preds[i])])
            #_pred = [idx2label[elem] for elem in preds] #idx2label[elem]
            #print(_pred)
            result.append([data['id'],_pred ])

        

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入第一列資料
        writer.writerow(['id', 'tags'])
        # 寫入剩下資料
        for _id,label in result:
            label = convert2str(label)
            writer.writerow([_id[0], label])

def convert2str(label:List[str]):
    s = ''
    _len = len(label)
    times=0
    for lab in label:
        s +=lab
    if(times != (_len - 1)):
        s+=" "
    times+=1
    print(s)
    return s

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json",
        #required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./model_state_dict_slot.pt",
        #required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
