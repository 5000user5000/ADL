import numpy as np
import csv
import torch
from argparse import ArgumentParser, Namespace

from transformers import (
    AutoModelForMultipleChoice, 
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    HfArgumentParser,
    DefaultDataCollator,
    EvalPrediction,
    Trainer,
    TrainingArguments
)

from datasets import load_dataset
import json 
from typing import Optional, Union ,List
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from pathlib import Path
from trainer import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions




def main(args):
    
    cs_pretrained_name = "./ckpt/CS_pretrain"
    qa_pretrained_name = "./ckpt/QA_pretrain"
    
    # fix random seed
    torch.manual_seed(777)
    
    # pretrained model and tokenizer
    cs_model = AutoModelForMultipleChoice.from_pretrained(
        cs_pretrained_name
    )
    cs_tokenizer = AutoTokenizer.from_pretrained(
        cs_pretrained_name
    )
    
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        qa_pretrained_name
    )
    qa_tokenizer = AutoTokenizer.from_pretrained(
        qa_pretrained_name
    )
    
    
    

    
    test_file_path = str(args.testing_file) #不加str()會報錯
    context_file_path = str(args.context_file)

    
    
    # read file
    with open(test_file_path, 'r', encoding='utf-8') as context_f:
            test_context: List = json.load(context_f)
    with open(context_file_path, 'r', encoding='utf-8') as context_f:
            context_context: List = json.load(context_f)



    test_dataset = load_dataset("json",data_files=test_file_path)
    

    # 將問題和選項 預處理
    def preprocess_function(examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        first_sentences = [[context] * 4 for context in examples["question"]]
        # Grab all second sentences possible for each context.
        question_headers = examples["paragraphs"]
        second_sentences = [[context_context[idx] for idx in header] for i, header in enumerate(question_headers)]
        
        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        # Tokenize
        tokenized_examples = cs_tokenizer(first_sentences, second_sentences, truncation=True)
        
        # Un-flatten
        return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


    # 把資料 批次丟入(map)給預處理 , drop_last_batch可以不寫 , 預設batch 1000 , last batch不會讀取的樣子
    encoded_datasets = test_dataset.map(preprocess_function, batched=True , drop_last_batch=False)

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
            #label_name = "label" if "label" in features[0].keys() else "labels"
            #print("==========")
            #print(features[0].keys())
            #labels = [feature.pop(label_name) for feature in features]
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
            #batch["labels"] = torch.tensor(labels, dtype=torch.int64)
            return batch

    
    
    
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


    # initialize trainer
    cs_trainer = Trainer(
        model=cs_model,
        tokenizer=cs_tokenizer,
        data_collator=DataCollatorForMultipleChoice(cs_tokenizer),
        compute_metrics=compute_metrics,
    )

    # context selection prediction
    print('Begin context selection...')
    result = cs_trainer.predict(encoded_datasets["train"])
    context_preds = np.argmax(result[0], axis=1) #預測出的標籤

    pred = test_context

    for i in range(len(test_context)):
        idx =context_preds[i]
        parag_idx =  test_context[i]["paragraphs"][idx]
        pred[i]["context"] = context_context[parag_idx]
        del pred[i]["paragraphs"]
    
    cs_pred_save_path = "./data/cs_pred.json"

    #將預測的label寫入
    with open(cs_pred_save_path, 'w', encoding='utf-8') as f:
        json.dump(pred, f)

    #################### qa ####################
    cs_pred_path = "./data/cs_pred.json"
    qa_dataset = load_dataset("json",data_files=cs_pred_path)
    

    def prepare_test_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        pad_on_right = qa_tokenizer.padding_side == "right"
        tokenized_examples = qa_tokenizer(
            examples['question' if pad_on_right else 'context'],
            examples['context' if pad_on_right else 'question'],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    column_names = qa_dataset['train'].column_names
    qa_test_dataset = qa_dataset['train'].map(
        prepare_test_features,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=False
    )

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
        )
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        return EvalPrediction(predictions=formatted_predictions, label_ids=None)

    data_collator = DefaultDataCollator() 
    qa_trainer = QuestionAnsweringTrainer(
        model=qa_model,
        data_collator=data_collator,
        tokenizer=qa_tokenizer,
        post_process_function=post_processing_function,
    )
    print("Begin question answering...")
    result = qa_trainer.predict(
        predict_dataset=qa_test_dataset,
        predict_examples=qa_dataset['train']
    )
    
    output_file = str(args.pred_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer'])
        rows = []
        for pred in result.predictions:
            rows.append([pred["id"], pred["prediction_text"]])
        writer.writerows(rows)
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_file",
        type=Path,
        help="Directory to the context file.",
        required=True
    )
    parser.add_argument(
        "--testing_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Path to the predict file.",
        required=True
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)