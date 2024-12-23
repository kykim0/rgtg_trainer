#!/bin/bash

ref_model=kykim0/pythia-1b-tulu-v2-mix
rmodel=kykim0/pythia-1b-tulu-v2-mix-uf-rm
base_path=/home/hyunseoki/trainer_rgtg/checkpoints
pyfile=/home/hyunseoki/trainer_rgtg/scripts/eval_greedy.py



# kykim0/ultrafeedback_binarized_cleaned_20p

i=0
num_gpus=2
offset=2
ckpt_base=${ref_model}
exp_dir="sft_model"
exp_path=${base_path}/${exp_dir}
echo "Processing $(basename ${exp_path})"

eval_dir=${exp_path}/eval_results
mkdir -p ${eval_dir}


CUDA_VISIBLE_DEVICES=2 python3 ${pyfile} \
    --llm ${ckpt_base} \
    --rm ${rmodel} \
    --ref_llm ${ref_model} \
    --log_file ${eval_dir}/ref_model.jsonl \
    --batch_size 4 \
    --max_new_tokens 512 \
    --split test_prefs &
