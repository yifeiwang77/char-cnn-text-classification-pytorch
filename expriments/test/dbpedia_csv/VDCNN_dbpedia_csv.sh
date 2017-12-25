
#!/usr/bin/env bash
cd ../../../
dataset="dbpedia_csv" 
epoch_size=200
batch_size=128
dropout=0.5
model="VDCNN"

python test.py      --model ${model} \
                    --test-path "data/${dataset}/test.csv" \
                    --model-path "checkpoints/${model}/${dataset}/Char${model}_best.pth.tar" 
                    --model ${model} \
                    --batch-size ${batch_size} \
                    --dropout ${dropout} \
