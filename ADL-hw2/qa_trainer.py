import json
from typing import List
from pathlib import Path
from transformers import DefaultDataCollator , AutoModelForQuestionAnswering, TrainingArguments, Trainer , AutoTokenizer
from datasets import load_dataset


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
with open(context_file_path, 'r', encoding='utf-8') as context_f:
            context_context: List = json.load(context_f)

# 寫入label (也就是在paragraphs的index)
def preprocess_data(context):
    for i in range(len(context)):
        del context[i]["paragraphs"]
        idx = context[i]["relevant"]
        context[i]["context"] = context_context[idx]
        context[i]["answers"] = {}
        context[i]["answers"]["text"] = [context[i]["answer"]["text"]]
        context[i]["answers"]["answer_start"] = [context[i]["answer"]["start"]]
        del context[i]["relevant"]
        del context[i]["answer"]

preprocess_data(train_context)
preprocess_data(valid_context)



# 將新的資料寫入到新的json(且要建置檔案) prep表示預處理過的
preprocess_train_path = './data/qa_prep_train.json'
preprocess_valid_path = './data/qa_prep_valid.json'
myfile = Path(preprocess_train_path)
myfile.touch(exist_ok=True)
myfile2 = Path(preprocess_valid_path)
myfile2.touch(exist_ok=True)

with open(preprocess_train_path, 'w') as f:
    json.dump(train_context, f)
with open(preprocess_valid_path, 'w') as f:
    json.dump(valid_context, f)



train_dataset = load_dataset("json",data_files=preprocess_train_path)
valid_dataset = load_dataset("json",data_files=preprocess_valid_path)


raw_dataset = train_dataset
raw_dataset["valid"] = valid_dataset["train"] #預設都是train
#print(raw_dataset)


model_name = "hfl/chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs




tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)


data_collator = DefaultDataCollator()


model = AutoModelForQuestionAnswering.from_pretrained(model_name)





training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    gradient_accumulation_steps=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# 儲存model等數據
pt_save_directory = "./qa_pt_save_pretrained"
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)
trainer.save_model(pt_save_directory) 


