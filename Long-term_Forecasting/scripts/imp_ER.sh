export CUDA_VISIBLE_DEVICES=0

seq_len=18
model=GPT4TS

for pred_len in 12
do
for percent in 100
do

python main.py \
    --root_path /home/dan/NeurIPS2023-One-Fits-All/dataset/Bond/ \
    --data_path imp_ER.csv \
    --model_id imp_ER_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 9 \
    --pred_len $pred_len \
    --batch_size 2048 \
    --learning_rate 0.0001 \
    --train_epochs 500 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 2 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 8759 \
    --c_out 8759 \
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
    --target 'ER'
done
done

