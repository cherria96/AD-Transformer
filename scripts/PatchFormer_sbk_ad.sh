if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

label_len=7
# patch_len=7
# stride=1
model=Patchformer

root_path_name=./data/
data_path_name=sbk_ad_selected.csv
dataset_name=sbk_ad_selected
model_id_name=Patchformer_simulated
if [ ! -d "./logs/"$dataset_name ]; then
    mkdir ./logs/$dataset_name
fi

random_seed=100
START=1
END=14
for pred_len in 14; do 
    for seq_len in 14; do
        for patch_len in 7; do
            for stride in 1; do
                patchnum=$(( ($seq_len - $patch_len) / $stride + 1 ))
                
                python -u main.py \
                --random_seed $random_seed \
                --is_training 1 \
                --root_path $root_path_name \
                --data_path $data_path_name \
                --model_id $model_id_name'_'$patch_len'_'$stride'_'$patchnum'_'$seq_len'_'$pred_len \
                --model $model \
                --data $dataset_name \
                --features MS \
                --n_subs 2 \
                --seq_len $seq_len \
                --label_len $label_len \
                --pred_len $pred_len \
                --patch_len $patch_len \
                --stride $stride \
                --enc_in 11 \
                --dec_in 11 \
                --c_out 11 \
                --d_model 64 \
                --n_heads 4 \
                --embed_type 0\
                --gpu 0\
                --kernel_size 7 \
                --decoder_mode future_subs \
                --embed fixed \
                --series_decomposition \
                --revin \
                --output_attention \
                --itr 1 --batch_size 32 --learning_rate 0.001 --patience 3 >logs/$dataset_name/'_'$model_id_name'_'$seq_len'_'$pred_len'_'$patch_len'_'$stride.log 
            done
        done
    done
done
