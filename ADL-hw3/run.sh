python3 ./run_summarization.py \
--model_name_or_path ./ckpt \
--do_predict \
--test_file ./data/public.jsonl \
--output_dir ./ \
--per_device_eval_batch_size=8 \
--predict_with_generate \
--text_column maintext \