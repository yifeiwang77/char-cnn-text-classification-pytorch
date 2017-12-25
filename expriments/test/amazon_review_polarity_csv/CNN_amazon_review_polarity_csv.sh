
#!/usr/bin/env bash
cd ../../../
dataset="amazon_review_polarity_csv"
epoch_size=200
batch_size=128
dropout=0.5
model="CharCNN"

python test.py      --model ${model} \
                    --model-path "checkpoints/${model}/${dataset}/${model}_best.pth.tar" 
                    --test-path "data/${dataset}/test.csv" 
                    --model ${model} \
                    --onehot \
                    --batch-size ${batch_size} \
                    --dropout ${dropout} \
