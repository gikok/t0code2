python single_task_fine_tune_csv.py \
    --dataset_name "super_glue" \
    --dataset_config_name "wic" \
    --template_name "grammar_homework" \
    --model_name_or_path "/home/transformers/" \
    --output_dir "/home/gikok/output" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --target_max_length 256 \
    --parallelize
    