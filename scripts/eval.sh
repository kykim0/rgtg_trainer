#!/bin/bash

ref_model=kykim0/pythia-1b-tulu-v2-mix
rmodel=kykim0/pythia-1b-tulu-v2-mix-uf-rm
base_path=/home/hyunseoki/trainer_rgtg/checkpoints
pyfile=/home/hyunseoki/trainer_rgtg/scripts/eval_greedy.py

declare -a exp_dirs=(
    # "iter1_1.0e-6_beta1.0_adamw_torch_rgtg1.5_kl_acc32_reward_test/kl__1"
    # "iter1_1.0e-6_beta1.0_adamw_torch_rgtg1.5_kl_acc4_reward_test/kl__1"
    # "iter1_1.0e-6_beta0.05_adamw_torch_rgtg1.5_kl_acc32_reward_test/kl__1"
    "iter1_1.0e-6_beta1.0_adamw_torch_rgtg0.5_kl_acc32_reward_nolength/kl__1"
    "iter1_1.0e-6_beta1.0_adamw_torch_rgtg1.5_kl_acc32_reward_nolength/kl__1"
    "iter1_1.0e-6_beta1.0_adamw_torch_rgtg1.5_kl_acc32_reward_nolength_adaptweight/kl__1"
)

# kykim0/ultrafeedback_binarized_cleaned_20p

i=0
num_gpus=2
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
