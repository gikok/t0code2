python train.py \
    --dataset_name "prompts2opts_001.parquet.gzip" \
    --model_name_or_path "bigscience/T0_3B" \
    --output_dir "/home/gikok/output" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --freeze_encoder \
    --learning_rate 1e-4 
    
