#!/bin/bash

# device=7
ref_model=kykim0/pythia-1b-tulu-v2-mix
rmodel=kykim0/pythia-1b-tulu-v2-mix-uf-rm
base_path=/home/kykim/dev/local/trainer_rgtg/output/pythia-1b-pythia-1b-rm-pilot
pyfile=/home/kykim/dev/local/trainer_rgtg/etc/eval_greedy.py

declare -a exp_dirs=(
    "b64-lr3e-06-kl0.05-vf0.1-wr0-l16-512-pe4-s42"
    "b128-lr3e-06-kl0.05-vf0.1-wr0-l16-512-pe4-s42"
)

# kykim0/ultrafeedback_binarized_cleaned_20p

i=0
num_gpus=4
offset=0
for exp_dir in "${exp_dirs[@]}"; do
    exp_path=${base_path}/${exp_dir}
    echo "Processing $(basename ${exp_path})"

    eval_dir=${exp_path}/eval_results
    mkdir -p ${eval_dir}

    for ckpt_dir in $(ls -d ${exp_path}/checkpoint-*); do
        device=$(( i % ${num_gpus} + ${offset}))

        ckpt_base=$(basename ${ckpt_dir})

        CUDA_VISIBLE_DEVICES=${device} python3 ${pyfile} \
            --llm ${ckpt_dir} \
            --rm ${rmodel} \
            --ref_llm ${ref_model} \
            --log_file ${eval_dir}/${ckpt_base}.jsonl \
            --batch_size 4 \
            --max_new_tokens 512 \
            --split test_prefs &
        (( ++i ))
        if (( i % ${num_gpus} == 0 )); then
            wait
            echo "Waited ${i}"
        fi
    done
done
