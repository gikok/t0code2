python train.py \
    --dataset_name "prompts004.parquet.gzip" \
    --model_name_or_path "bigscience/T0_3B" \
    --output_dir "/home/gikok/output" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --freeze_encoder \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --parallelize
    
