import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader
import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
import csv


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    idx2label = {idx: intent for intent, idx in intent2idx.items()}
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: create DataLoader for test dataset
    dataloader_test = DataLoader(dataset=dataset, batch_size=args.batch_size ,shuffle=True,collate_fn=dataset.collate_fn_test)

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
    # TODO: predict dataset
    result = []
    for num,data in enumerate(dataloader_test):  
            input =  data['input'].to(device)
            output = model(input)
            _, preds = torch.max(output, 1)
            for i in range(len(preds)): #batch_size
                result.append([data['id'][i],idx2label[preds[i].item()] ])

        

    # TODO: write prediction to file (args.pred_file)
    with open('intent_output.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入第一列資料
        writer.writerow(['id', 'intent'])
        # 寫入剩下資料
        for _id,label in result:
            writer.writerow([_id, label])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
        #required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/model_state_dict.pt",
        #required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
