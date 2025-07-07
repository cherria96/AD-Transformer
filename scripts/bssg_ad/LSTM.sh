if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

label_len=7
# patch_len=7
# stride=1
model=LSTM

root_path_name=./data/
data_path_name=bssg_ad.csv
dataset_name=bssg_ad
model_id_name=LSTM

if [ ! -d "./logs/"$dataset_name ]; then
    mkdir ./logs/$dataset_name
fi

random_seed=100
START=1
END=5
for pred_len in $(seq $START $END); do 
    for seq_len in 14 28 42 56; do
        python -u main.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model \
        --data $dataset_name \
        --features MS \
        --n_subs 3 \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --enc_in 10 \
        --dec_in 10 \
        --c_out 1 \
        --d_model 64 \
        --n_heads 4 \
        --embed_type 0\
        --gpu 0\
        --kernel_size 7 \
        --decoder_mode default \
        --revin \
        --embed fixed \
        --itr 1 --batch_size 32 --learning_rate 0.001 --patience 3 >logs/$dataset_name/'_'$model_id_name'_'$seq_len'_'$pred_len'_'$patch_len'_'$stride.log 
    done
done
