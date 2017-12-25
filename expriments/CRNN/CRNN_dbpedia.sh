#!/usr/bin/env bash
cd ../../
dataset="dbpedia_csv"
epoch_size=200
batch_size=128
dropout=0.5
kernel_num=128

python train_CRNN.py --train-path "data/${dataset}/train.csv" \
                    --val-path "data/${dataset}/test.csv" \
                    --epochs ${epoch_size} \
                    --batch-size ${batch_size} \
                    --dropout ${dropout} \
                    --lr 0.0005 \
                    --dynamic-lr \
                    --decay-factor 0 \
                    --save-folder "checkpoints/CharCRNN/${dataset}" 