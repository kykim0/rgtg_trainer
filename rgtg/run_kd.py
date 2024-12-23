"""KD trainer."""

import logging
import os
import sys

from accelerate import PartialState
from alignment import (
    DataArguments,
    get_datasets,
    get_quantization_config,
    H4ArgumentParser,
    ModelArguments,
    KDConfig,
)
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from kd_trainer import KDTrainer

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((KDConfig, ModelArguments, DataArguments))
    config, model_config, data_config = parser.parse()

    # Set seed for reproducibility
    set_seed(config.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = config.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"KDConfig {config}")
    logger.info(f"ModelConfig {model_config}")
    logger.info(f"DataConfig {data_config}")

    num_gpus = torch.cuda.device_count()
    per_device_batch = config.per_device_train_batch_size
    grad_accum_steps = config.gradient_accumulation_steps
    batch_size = num_gpus * per_device_batch * grad_accum_steps
    run_name = "-".join([
        f"b{batch_size}",
        f"lr{config.learning_rate}",
        f"a{config.teacher_mixed_alpha or 0}",
        f"l{config.min_response_length or 0}-{config.response_length}",
        f"e{config.num_train_epochs}",
        f"pe{config.num_ppo_epochs}",
        f"ln{int(config.length_norm)}",
        f"ss{int(config.single_step_reg)}",
        f"s{config.seed}",
    ])
    if "wandb" in config.report_to:
        config.tracker_kwargs = {"wandb": {"name": run_name}}
    config.run_name = run_name
    config.output_dir = os.path.join(config.output_dir, config.run_name)

    ###################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_config.torch_dtype if model_config.torch_dtype in ["auto", None] else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if config.gradient_checkpointing else True,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    assert(tokenizer.chat_template is not None)

    # Set up models.
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_path, **model_kwargs)
    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_path, **model_kwargs)

    #########
    # Dataset
    #########
    def dataset_map_fn(example, tokenizer):
        messages = [{"role": "user", "content": example["prompt"]}]
        query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        example_out = {
            "input_ids": tokenizer.encode(query),
        }
        return example_out

    # We retrieve the dataloader by calling the `build_dataset` function.
    # Compute that only on the main process for faster data processing.
    # See: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        column_names = ["prompt", "prompt_id", "chosen", "rejected", "messages", "score_chosen", "score_rejected", "source"]
        raw_datasets = get_datasets(data_config, splits=data_config.dataset_splits, columns_to_keep=column_names, shuffle=True)
        raw_datasets = raw_datasets.map(
            dataset_map_fn,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=column_names,
            num_proc=data_config.preprocessing_num_workers,
        )
        raw_datasets.set_format(type="torch")
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["test"]

    ##########
    # Training
    ##########
    trainer = KDTrainer(
        config=config,
        tokenizer=tokenizer,
        model=model,
        teacher_model=teacher_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()


if __name__ == "__main__":
    main()
