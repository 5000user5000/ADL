python3 ./run_summarization.py \
--model_name_or_path google/mt5-small \
--do_train \
--do_eval \
--train_file ./data/train.jsonl \
--validation_file ./data/public.jsonl \
--source_prefix "summarize: " \
--output_dir ./ckpt/ \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--eval_accumulation_steps=128 \
--overwrite_output_dir \
--predict_with_generate \
--text_column maintext \
--summary_column title \
--learning_rate 1e-3 \
--warmup_ratio 0.1 \
#--report_to wandb \
#--run_name ADL3-1119 \