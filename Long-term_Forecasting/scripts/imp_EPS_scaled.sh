export CUDA_VISIBLE_DEVICES=0

seq_len=18
model=GPT4TS

for which_firm in 2 3 13 18 19 21 22 23 29 34 35 38 40 46 47 49 51 58 60 63 64 66 79 83 85 86 88 89 93 96 101 102 117 119
do
for pred_len in 1
do
for percent in 100
do

python main.py \
    --root_path /home/dan/NeurIPS2023-One-Fits-All/dataset/Data/ \
    --data_path df_EPS_CHAR.csv \
    --model_id imp_EPS_scaled_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data eps \
    --seq_len $seq_len \
    --label_len 9 \
    --pred_len $pred_len \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --train_epochs 500 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 2 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 13149 \
    --c_out 13149 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --target 'value'\
    --which_firm $which_firm \

done
done
done

