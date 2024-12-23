#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
from dataclasses import asdict, fields
import logging
import os
import sys

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    PPOArguments,
    LlamaRewardModel,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
import torch
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    set_seed,
)
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from trl.core import LengthSampler

from fastchat_eval import run_eval

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, PPOArguments))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    num_gpus = torch.cuda.device_count()
    mini_batch, grad_accum_steps = training_args.mini_batch_size, training_args.gradient_accumulation_steps
    effective_batch_size = num_gpus * mini_batch * grad_accum_steps
    run_name = "-".join([
        f"b{effective_batch_size}",
        f"e{training_args.ppo_epochs}",
        f"s{training_args.steps}",
        f"lr{training_args.learning_rate}",
        f"{training_args.kl_penalty}{training_args.init_kl_coef}",
        f"gam{training_args.gamma}",
        f"lam{training_args.lam}",
        f"vf{training_args.vf_coef}",
        f"t{training_args.temperature}",
    ])
    task_name = training_args.task_name
    run_name = f"{run_name}-{task_name}" if task_name else run_name
    if training_args.log_with == "wandb":
        training_args.tracker_kwargs = {"wandb": {"name": run_name}}

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.tracker_project_name,
        run_name,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits, token=training_args.hub_token, shuffle=True)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    def dataset_map_fn(example, tokenizer):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": example["prompt"]},
        ]
        example["original_prompt"] = example["prompt"]
        example["query"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        example["input_ids"] = tokenizer.encode(example["query"])
        return example

    raw_datasets = raw_datasets.map(
        dataset_map_fn,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
    )
    raw_datasets.set_format(type="torch")
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    #########################
    # Instantiate PPO trainer
    #########################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        peft_config=get_peft_config(model_args),
        **model_kwargs,
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    logger.info("*** Model loaded! ***")

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    ppo_args_dict = asdict(training_args)
    ppo_config_args = {
        field.name: ppo_args_dict[field.name]
        for field in fields(PPOConfig) if field.name in ppo_args_dict
    }
    ppo_config_args['seed'] = training_args.seed
    ppo_config_args['project_kwargs'] = {'logging_dir': training_args.output_dir}
    ppo_config = PPOConfig(**ppo_config_args)
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )
    accelerator = ppo_trainer.accelerator

    rmodel_name = training_args.reward_model
    rmodel_tokenizer_cls, rmodel_cls = AutoTokenizer, AutoModelForSequenceClassification
    if rmodel_name == "openbmb/UltraRM-13b":
        rmodel_tokenizer_cls, rmodel_cls = LlamaTokenizer, LlamaRewardModel
    rmodel_tokenizer = rmodel_tokenizer_cls.from_pretrained(rmodel_name)
    rmodel_quant_config = BitsAndBytesConfig(load_in_8bit=True)
    rmodel = rmodel_cls.from_pretrained(
        rmodel_name,
        quantization_config=rmodel_quant_config,
    )
    rmodel.eval()
    if getattr(rmodel_tokenizer, "pad_token", None) is None:
        rmodel_tokenizer.pad_token = rmodel_tokenizer.eos_token
        rmodel.config.pad_token_id = rmodel_tokenizer.eos_token_id

    ###############
    # Training loop
    ###############
    # See https://huggingface.co/docs/transformers/main_classes/text_generation.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": training_args.output_max_length,
    }
    output_min_length = training_args.output_min_length
    output_max_length = training_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    incompletion_penalty = torch.tensor(training_args.incompletion_penalty, dtype=torch.float)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=training_args.eval_batch_size,
        collate_fn=collator,
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)

    def run_generation(batch, mode):
        query_tensors = batch["input_ids"]

        # Get responses from the base model.
        length_sampler = output_length_sampler if mode == "train" else None
        temperature = training_args.temperature if mode == "train" else training_args.eval_temperature
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            length_sampler=length_sampler,
            temperature=temperature,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # Compute the reward scores.
        texts = [q + r + "\n" for q, r in zip(batch["query"], batch["response"])]
        with torch.no_grad():
            rmodel_inputs = rmodel_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            rmodel_outputs = rmodel(**rmodel_inputs)
            if not rmodel_name == "openbmb/UltraRM-13b":
                rmodel_outputs = rmodel_outputs.logits
            rewards = [
                r.squeeze().to(dtype=torch.float) if text.endswith(tokenizer.eos_token + "\n") else incompletion_penalty
                for text, r, in zip(texts, rmodel_outputs)
            ]

        return response_tensors, rewards

    def local_eval(dataloader, out_fname):
        eval_out = []
        for batch in tqdm(
            dataloader,
            desc=f"Local eval ({accelerator.local_process_index}): ",
        ):
            _, rewards = run_generation(batch, "eval")
            for q, r, reward in zip(batch["query"], batch["response"], rewards):
                eval_out.append({"query": q, "response": r, "reward": reward.item()})

        eval_out = accelerator.gather_for_metrics(eval_out)

        if accelerator.is_main_process:
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
            with open(out_fname, "w", newline="", encoding="UTF-8") as csvfile:
                fieldnames = ["query", "response", "reward"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in eval_out:
                    writer.writerow(row)

        accelerator.wait_for_everyone()

    eval_out_all = []
    for step, batch in tqdm(
        enumerate(ppo_trainer.dataloader),
        desc="Train step: ",
        disable=not accelerator.is_local_main_process,
        total=min(len(ppo_trainer.dataloader), ppo_config.steps),
    ):
        if step >= ppo_config.steps: break

        # Run periodic evals.
        if training_args.eval_freq and step % training_args.eval_freq == 0:
            eval_fname = f"{training_args.output_dir}/eval_mt-bench_{step}.jsonl"
            run_eval(accelerator, model, tokenizer, answer_file=eval_fname,
                     max_new_token=generation_kwargs["max_new_tokens"])
            local_eval_fname = f"{training_args.output_dir}/eval_local_{step}.csv"
            local_eval(eval_dataloader, local_eval_fname)

        query_tensors = batch["input_ids"]
        response_tensors, rewards = run_generation(batch, "train")

        # Run PPO step.
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if (accelerator.is_main_process and
            training_args.save_freq and
            (step + 1) % training_args.save_freq == 0):
            save_dir = f"{training_args.output_dir}/step_{step + 1}"
            ppo_trainer.save_pretrained(save_dir)
            torch.save(ppo_trainer.optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pth'))

    logger.info("*** Training complete ***")

    # Run final evals.
    eval_fname = f"{training_args.output_dir}/eval_mt-bench_last.jsonl"
    run_eval(accelerator, model, tokenizer, answer_file=eval_fname,
             max_new_token=generation_kwargs["max_new_tokens"])
    local_eval_fname = f"{training_args.output_dir}/eval_local_last.csv"
    local_eval(eval_dataloader, local_eval_fname)

    ##################################
    # Save model and create model card
    ##################################
    if accelerator.is_main_process:
        save_dir = f"{training_args.output_dir}/last"
        ppo_trainer.save_pretrained(save_dir)
        torch.save(ppo_trainer.optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pth'))
        if training_args.push_to_hub is True:
            ppo_trainer.push_to_hub()

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
