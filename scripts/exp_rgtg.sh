set -e
set -x

export OMP_NUM_THREADS=2

LEARNING_RATE="1.0e-6"
ITER="1"
BETA="1.0"
LOSS_TYPE="kl"
OPTIM="adamw_torch"
# PREF="sppo_score"
NUM=1
MODEL="kykim0/pythia-1b-tulu-v2-mix"
RMMODEL="kykim0/pythia-1b-tulu-v2-mix-uf-rm"
DATASET="kykim0/ultrafeedback_binarized_cleaned_20p"

BATCH_SIZE=1
ACCUMULATE=32
RGTG_WEIGHT="2.5"

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --learning_rate)
        LEARNING_RATE="$2"
        shift
        ;;
    --beta)
        BETA="$2"
        shift
        ;;
    --optim)
        OPTIM="$2"
        shift
        ;;
    --output_dir)
        OUTPUT_DIR="$2"
        shift
        ;;
    --iter)
        ITER="$2"
        shift
        ;;
    --loss_type)
        LOSS_TYPE="$2"
        shift
        ;;
    --prefix)
        PREF="$2"
        shift
        ;;
    --model)
        MODEL="$2"
        shift
        ;;
    --rm_model)
        RMMODEL="$2"
        shift
        ;;
    --dataset)
        DATASET="$2"
        shift
        ;;
    --num)
        NUM="$2"
        shift
        ;;
    --batch_size)
        BATCH_SIZE="$2"
        shift
        ;;
    --accumulate)
        ACCUMULATE="$2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

PREF="${PREF}_${NUM}"

LEVEL1="iter${ITER}_${LEARNING_RATE}_beta${BETA}_${OPTIM}_rgtg${RGTG_WEIGHT}_${LOSS_TYPE}_acc${ACCUMULATE}_rgtg"
LEVEL2="${LOSS_TYPE}_${PREF}"

OUTPUT_DIR="checkpoints/${LEVEL1}/${LEVEL2}"
log_file="iter${ITER}_${LEARNING_RATE}_${BETA}_${OPTIM}_${LOSS_TYPE}_${PREF}_rgtg${RGTG_WEIGHT}_acc${ACCUMULATE}_rgtg"

dataset_name=$(echo "$DATASET" | cut -d '/' -f2)
new_config_file="recipes/rgtg/exp_new.yaml"

# Copy the original configuration file to the new one
# cp config/exp.yaml "$new_config_file"

python3 scripts/update_dataset.py --dataset $DATASET --config "$new_config_file" >"$log_file.log"

echo "logging to $log_file.log"

export WANDB_DISABLED=false
export CUDA_VISIBLE_DEVICES=4,5
export ACCELERATE_LOG_LEVEL=info
# --main_process_port ${port} \
#  python run_trainer.py "$new_config_file" \
#    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    # --config_file accelerate_configs/multi_gpu.yaml \
accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero2.yaml \
    --main_process_port 4930 \
    --num_processes 2 \
    rgtg/run_trainer_rgtg.py "$new_config_file" \
    --learning_rate=$LEARNING_RATE \
    --beta=$BETA \
    --optim="$OPTIM" \
    --save_strategy="steps" \
    --save_steps=0.1 \
    --output_dir="$OUTPUT_DIR" \
    --run_name="rgtg" \
    --loss_type=$LOSS_TYPE \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$ACCUMULATE \
    --model_name_or_path=$MODEL \
    --rm_model_name_or_path=$RMMODEL \
    --num_train_epochs=$NUM \
    --rgtg_weight=$RGTG_WEIGHT \
    --gradient_checkpointing=false \
    --topk=10 
    # --use_ref=true
# 2>&1 | tee "${log_file}.log"
#bash scripts/pipeline.sh --model $MODEL --iter $i --dataset "synthetic_data_gemma-2-9b-it-sppo-iter${i}_score" --output_dir $OUTPUT_DIR --num 1 --batch_size 4 --accumulate 2
