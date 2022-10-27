python -m torch.distributed.run --nproc_per_node=8 --use_env train_pl.py \
    --dataset_name "seven_prompts001.parquet.gzip" \
    --eval_name "miniprompts005_eval.parquet.gzip" \
    --model_name_or_path "bigscience/T0_3B" \
    --output_dir "/home/gikok/seven_prompts" \
    --num_train_epochs 25 \
    --max_length 512 \
    --target_max_length 256 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --freeze_encoder \
    --gradient_accumulation_steps 32 \
    --learning_rate 0.0001 \
    --weight_decay 0.0 \
    --parallelize
    
