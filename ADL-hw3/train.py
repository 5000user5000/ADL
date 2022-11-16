from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

tokenizer = AutoTokenizer.from_pretrained("t5-small")

prefix = "title: "



def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["maintext"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["title"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_dataset = load_dataset("json",data_files="./data/train.jsonl") #jsonl直接用即可
test_dataset  = load_dataset("json",data_files="./data/public.jsonl")

train_dataset["test"] = test_dataset["train"]

tokenized_train_dataset= train_dataset.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset["train"],
    eval_dataset=tokenized_train_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()