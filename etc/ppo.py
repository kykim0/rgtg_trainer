
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator, PartialState
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available


import model_training.models.reward_model  # noqa: F401 (registers reward model for AutoModel loading)

from utils import get_datasets


tqdm.pandas()


@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    dataset_num_proc: Optional[int] = field(default=4, metadata={"help": "num data processes"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})


def main():
    parser = HfArgumentParser((ScriptArguments, PPOConfig))
    args, ppo_config = parser.parse_args_into_dataclasses()

    # TODO(kykim): Do a better job.
    ppo_config.model_name = "kykim0/OLMo-1B-SFT-hf"
    ppo_config.query_dataset = "allenai/ultrafeedback_binarized_cleaned"
    ppo_config.reward_model = "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"

    # Set seed before initializing value head for deterministic eval.
    set_seed(ppo_config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    if not args.use_peft:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            ppo_config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        )
        device_map = None
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
        ref_model = None
        # Copy the model to each device.
        device_map = {"": Accelerator().local_process_index}

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id


    def dataset_map_fn(example, tokenizer):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": example["prompt"]},
        ]
        example["original_prompt"] = example["prompt"]
        example["query"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        example["input_ids"] = tokenizer.encode(example["query"])
        return example


    # We retrieve the dataloader by calling the `build_dataset` function.
    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        data_mixer = {ppo_config.query_dataset: 1.0}
        columns = ["prompt", "prompt_id", "chosen", "rejected", "messages", "score_chosen", "score_rejected", "source"]
        raw_datasets = get_datasets(data_mixer, splits=["train_gen", "test_gen"], columns_to_keep=columns, shuffle=True)
        raw_datasets = raw_datasets.map(
            dataset_map_fn,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=4,
        )
        raw_datasets.set_format(type="torch")
        # TOOD(kykim): Fix the hard-coded split names.
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["test"]


    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}


    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=train_dataset, data_collator=collator)

    # Ensure that the device is the same as that of the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        if is_xpu_available():
            device = "xpu:0"
        elif is_npu_available():
            device = "npu:0"
        else:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    reward_model_name = ppo_config.reward_model
    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            rtokenizer = AutoTokenizer.from_pretrained(reward_model_name)
            rmodel = AutoModelForSequenceClassification.from_pretrained(
                reward_model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
    else:
        rtokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        rmodel = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    # rtokenizer.to(device)

    # TODO(kykim): Double check.
    # Args to the `generate` function of the PPOTrainer.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
    }

    rm_template = "<|prompter|>{query}<|endoftext|><|assistant|>{response}<|endoftext|>"

    for _epoch, batch in tqdm(
        enumerate(ppo_trainer.dataloader),
        total=len(ppo_trainer.dataloader),
    ):
        query_tensors = batch["input_ids"]

        # Get responses.
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute rewards.
        texts = [rm_template.format(query=q, response=r) for q, r in zip(batch["query"], batch["response"])]
        with torch.no_grad():
            rmodel_inputs = rtokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            rmodel_outputs = rmodel(**rmodel_inputs)
            rewards = [
                r.squeeze().to(dtype=torch.float) if text.endswith(tokenizer.eos_token) else -1.0
                for text, r, in zip(texts, rmodel_outputs.logits)
            ]

        # Compute reference rewards if needed.
        # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        ref_texts = [rm_template.format(query=q, response=r) for q, r in zip(batch["query"], batch["ref_response"])]
        with torch.no_grad():
            rmodel_inputs = rtokenizer(ref_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            rmodel_outputs = rmodel(**rmodel_inputs)
            ref_rewards = [
                r.squeeze().to(dtype=torch.float) if text.endswith(tokenizer.eos_token) else -1.0
                for text, r, in zip(texts, rmodel_outputs.logits)
            ]
            batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
        # break


if __name__ == "__main__":
    main()
