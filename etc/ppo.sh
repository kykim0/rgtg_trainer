CUDA_VISIBLE_DEVICES=6,7 WANDB_PROJECT=ppo accelerate launch \
    --config_file /home/kykim/dev/local/RewardTraining/soft_finetune/ppo/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=2 \
    --main_process_port=1357 \
    /home/kykim/dev/local/RewardTraining/soft_finetune/ppo/ppo.py \
    --batch_size 8 \
    --mini_batch_size 8


# python ppo.py \
#     --learning_rate 3e-6 \
#     --output_dir output/test_ppo2 \
#     --per_device_train_batch_size 64 \
#     --gradient_accumulation_steps 1 \
#     --total_episodes 10000 \
#     --model_name_or_path kykim0/gemma-2b-ultrachat-sft \
#     --sft_model_path kykim0/gemma-2b-ultrachat-sft \
#     --reward_model_path Ray2333/Gemma-2B-rewardmodel-baseline \
#     --non_eos_penalty

# CUDA_VISIBLE_DEVICES=6,7 WANDB_PROJECT=rgtg accelerate launch \
#     --config_file /home/kykim/dev/local/RewardTraining/soft_finetune/ppo/accelerate_configs/deepspeed_zero3.yaml \
#     --num_processes=2 \
#     /home/kykim/dev/local/RewardTraining/soft_finetune/ppo/ppo.py \
#     --output_dir output/test_ppo \
#     --num_ppo_epochs 2 \
#     --num_mini_batches 1 \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --response_length 256 \
#     --total_episodes 10000 \
#     --model_name_or_path kykim0/OLMo-1B-SFT-hf \
#     --sft_model_path kykim0/OLMo-1B-SFT-hf \
#     --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
#     --local_rollout_forward_batch_size 1 \
#     --non_eos_penalty true \
#     --torch_dtype bfloat16 \
#     --report_to none
#     # --deepspeed3

# accelerate_configs/deepspeed_zero3.yaml
# --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
# --reward_model_path OpenAssistant/reward-model-deberta-v3-large-v2
# --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5
# --reward_model_path Ray2333/Gemma-2B-rewardmodel-baseline
# --model_name_or_path kykim0/OLMo-1B-SFT-hf
# --model_name_or_path kykim0/gemma-2b-ultrachat-sft
