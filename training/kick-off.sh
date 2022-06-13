python train.py \
    --dataset_name "prompts_001.gzip" \
    --model_name_or_path "/home/transformers2/" \
    --output_dir "/home/gikok/output" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --parallelize
    