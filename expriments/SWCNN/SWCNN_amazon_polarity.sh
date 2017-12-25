#!/usr/bin/env bash
cd ../../
dataset="amazon_review_polarity_csv"
epoch_size=200
batch_size=128
dropout=0.5

python train_SWCNN.py --train-path "data/${dataset}/train.csv" \
                    --val-path "data/${dataset}/test.csv" \
                    --epochs ${epoch_size} \
                    --batch-size ${batch_size} \
                    --dropout ${dropout} \
                    --lr 0.0005 \
                    --dynamic-lr \
                    --decay-factor 0 \
                    --save-folder "checkpoints/SWCNN/${dataset}" 