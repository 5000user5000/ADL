import sys
import os
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
import logging

import datasets
import torch
from transformers.trainer_utils import get_last_checkpoint

from datasets import load_dataset
import json 
from typing import Optional, Union ,List,Dict

from pathlib import Path
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy






logger = logging.getLogger(__name__)




#參數
model_name = "bert-base-chinese"
batch_size = 2
args = TrainingArguments(
    f"{model_name}-finetuned-swag",
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=8,
    push_to_hub=False,
)


# Setup logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# set model and tokenizer
model = AutoModelForMultipleChoice.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")



# data path
context_file_path = "./data/context.json"
train_file_path = "./data/train.json"
valid_file_path = "./data/valid.json"
test_file_path = "./data/test.json"

# read file
with open(train_file_path, 'r', encoding='utf-8') as context_f:
            train_context: List = json.load(context_f)
with open(valid_file_path, 'r', encoding='utf-8') as context_f:
            valid_context: List = json.load(context_f)

# 寫入label (也就是在paragraphs的index)
for i in range(len(train_context)):
    train_context[i]["label"] = train_context[i]["paragraphs"].index(train_context[i]["relevant"]) 

for i in range(len(valid_context)):
    valid_context[i]["label"] = valid_context[i]["paragraphs"].index(valid_context[i]["relevant"])


# 將新的資料寫入到新的json(且要建置檔案) prep表示預處理過的
preprocess_train_path = './data/prep_train.json'
preprocess_valid_path = './data/prep_valid.json'
myfile = Path(preprocess_train_path)
myfile.touch(exist_ok=True)
myfile2 = Path(preprocess_valid_path)
myfile2.touch(exist_ok=True)

with open(preprocess_train_path, 'w') as f:
    json.dump(train_context, f)
with open(preprocess_valid_path, 'w') as f:
    json.dump(valid_context, f)


# 以dataset形式讀取預處理後的json和test
train_dataset = load_dataset("json",data_files=preprocess_train_path)
valid_dataset = load_dataset("json",data_files=preprocess_valid_path)
test_dataset = load_dataset("json",data_files=test_file_path)

# 合併成一個dataset中 (其實data_files那裏就可以區分好了,詳細可看load_dataset官網)
raw_dataset = train_dataset
raw_dataset["valid"] = valid_dataset["train"] #預設都是train
raw_dataset["test"] = test_dataset["train"]
print(raw_dataset)

# 讀取context.json
with open(context_file_path, 'r', encoding='utf-8') as context_f:
            raw_context: List = json.load(context_f)

# 將問題和選項 預處理
def preprocess_function(examples):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["question"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["paragraphs"]
    second_sentences = [[raw_context[idx] for idx in header] for i, header in enumerate(question_headers)]
    
    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


# 把資料 批次丟入(map)給預處理 , drop_last_batch可以不寫 , 預設batch 1000 , last batch不會讀取的樣子
encoded_datasets = raw_dataset.map(preprocess_function, batched=True , drop_last_batch=False)


# collate

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        #print("==========")
        #print(features[0].keys())
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

# 我開頭為何要寫入預處理json就是這裡的問題,因為train只接受label這個key名,relevant這類不接受
accepted_keys = ["input_ids", "attention_mask", "label"]
features = [{k: v for k, v in encoded_datasets["train"][i].items() if k in accepted_keys} for i in range(10)]
batch = DataCollatorForMultipleChoice(tokenizer)(features)


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["valid"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

#開始train
trainer.train()