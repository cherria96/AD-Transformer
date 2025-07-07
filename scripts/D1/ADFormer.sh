if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

label_len=7
# patch_len=7
# stride=1
model=ADFormer

root_path_name=./data/
data_path_name=D1.csv
dataset_name=D1
model_id_name=ADformer
if [ ! -d "./logs/"$dataset_name ]; then
    mkdir ./logs/$dataset_name
fi

random_seed=100
START=1
END=14
for pred_len in $(seq $END); do 
    for seq_len in 14 28 42 56; do
        for patchnum in 4 8; do
            for stride in 1 2 3; do
                patch_len=$(( seq_len - (patchnum - 1) * stride ))
                # Ensure valid patch_len
                if [ $patch_len -gt 0 ]; then
                    remainder=$(( (seq_len - patch_len) % stride ))
                    if [ $remainder -eq 0 ]; then
                        echo "Running: seq_len=$seq_len, patchnum=$patchnum, stride=$stride, patch_len=$patch_len"
                        python -u main.py \
                        --random_seed $random_seed \
                        --is_training 1 \
                        --root_path $root_path_name \
                        --data_path $data_path_name \
                        --model_id $model_id_name'_'$patch_len'_'$stride'_'$patchnum'_'$seq_len'_'$pred_len \
                        --model $model \
                        --data $dataset_name \
                        --features MS \
                        --seq_len $seq_len \
                        --label_len $label_len \
                        --pred_len $pred_len \
                        --patch_len $patch_len \
                        --stride $stride \
                        --enc_in 14 \
                        --dec_in 14 \
                        --c_out 14 \
                        --d_model 32 \
                        --n_heads 4 \
                        --embed_type 0\
                        --gpu 0\
                        --kernel_size 7 \
                        --decoder_mode default \
                        --embed fixed \
                        --series_decomposition \
                        --revin \
                        --output_attention \
                        --itr 1 --batch_size 32 --learning_rate 0.001 --patience 3 >logs/$dataset_name/'_'$model_id_name'_'$seq_len'_'$pred_len'_'$patch_len'_'$stride.log 
                    fi
                fi
            done
        done
    done
done
