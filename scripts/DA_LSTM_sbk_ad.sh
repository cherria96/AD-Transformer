if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

label_len=7
# patch_len=7
# stride=1
model=DALSTM

root_path_name=./data/
data_path_name=sbk_ad_selected.csv
dataset_name=sbk_ad_selected
model_id_name=DALSTM_HRT

if [ ! -d "./logs/"$dataset_name ]; then
    mkdir ./logs/$dataset_name
fi

random_seed=100
END=14
for pred_len in $(seq $END); do 
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
        --enc_in 11 \
        --dec_in 11 \
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
