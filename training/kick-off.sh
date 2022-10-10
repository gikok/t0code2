python train.py \
    --dataset_name "miniprompts002.parquet.gzip" \
    --eval_name "miniprompts002_eval.parquet.gzip" \
    --model_name_or_path "bigscience/T0pp" \
    --output_dir "/home/gikok/big_128_1_v3" \
    --num_train_epochs 1000 \
    --max_length 512 \
    --target_max_length 256 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --freeze_encoder \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.001 \
    --weight_decay 0.0 \
    --parallelize
    
