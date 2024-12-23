#
# Adapted from https://github.com/huggingface/alignment-handbook

import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed
# def add_cols(feature, chosen_probs, chosen_probs_win, chosen_probs_lose):
#     feature['chosen_probs'] = chosen_probs
#     feature['chosen_probs_win'] = chosen_probs_win
#     feature['chosen_probs_lose'] = chosen_probs_lose
#     return feature

def prepare_inputs_for_generation_o(
    input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2]

        # Some generation methods already pass only the last input ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1

        input_ids = input_ids[:, remove_prefix_length:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

class RGTGTrainer(Trainer):
    r"""
    Initialize SPPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        rm_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. In SPPO, eta=1/beta. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, defaults to 0):
            The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report that should be between 0 and 0.5.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of loss to use. 'sppo' reproduces the SPPO algorithms. Other choices are explained as follows: `"sigmoid"` represents the default DPO loss,`"hinge"` loss from [SLiC](https://arxiv.org/abs/2305.10425) paper, `"ipo"` from [IPO](https://arxiv.org/abs/2310.12036) paper, or `"kto"` from the HALOs [report](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf).
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
    """

    _tag_names = ["trl", "rgtg"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        rm_model: Union[PreTrainedModel, nn.Module, str] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: str = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: str = None,
        ref_adapter_name: str = None,
        topk: int = 40,
        rgtg_weight: float = 1.0,
        window_size: int = 2,
        lookup_size: int = 1,
        use_ref: bool = False,
    ):
        self.topk = topk
        self.rgtg_weight = rgtg_weight
        self.window_size = window_size
        self.lookup_size = lookup_size
        self.rm_model = rm_model
        self.use_ref = use_ref
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the SPPOTrainer. But your model is already instantiated.")
        if self.use_ref:
            print('use_ref')
            ref_model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
            self.ref_model = ref_model
        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SPPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        
        if isinstance(rm_model, str):
            warnings.warn(
                "You passed a model_id to the SPPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            rm_model = AutoModelForSequenceClassification.from_pretrained(rm_model, **model_init_kwargs)
            self.rm_model = rm_model
        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            raise NotImplementedError
            # # if model is a peft model and we have a peft_config, we merge and unload it first
            # if isinstance(model, PeftModel):
            #     model = model.merge_and_unload()

            # if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            #     _support_gc_kwargs = hasattr(
            #         args, "gradient_checkpointing_kwargs"
            #     ) and "gradient_checkpointing_kwargs" in list(
            #         inspect.signature(prepare_model_for_kbit_training).parameters
            #     )

            #     preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

            #     if _support_gc_kwargs:
            #         preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

            #     model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)
            # elif getattr(args, "gradient_checkpointing", False):
            #     # For backward compatibility with older versions of transformers
            #     if hasattr(model, "enable_input_require_grads"):
            #         model.enable_input_require_grads()
            #     else:

            #         def make_inputs_require_grad(module, input, output):
            #             output.requires_grad_(True)

            #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # # get peft model with the given config
            # model = get_peft_model(model, peft_config)
            # if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
            #     peft_module_casting_to_bf16(model)
            #     # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
            #     self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpoiting, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a SPPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the SPPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 2048
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the SPPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 1024

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the SPPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 512

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # tokenize the dataset
        # print('=== before map', train_dataset.features)
        # chosen_probs = train_dataset['chosen_probs']
        # chosen_probs_win = train_dataset['chosen_probs_win']
        # chosen_probs_lose = train_dataset['chosen_probs_lose']
        # old_train_dataset = train_dataset
        # train_dataset = train_dataset.map(self.tokenize_row)
        # print('=== before add', train_dataset.features)
        # import pandas as pd
        # mid_dataset = pd.DataFrame(train_dataset)
        # mid_dataset['chosen_probs'] = chosen_probs
        # mid_dataset['chosen_probs_win'] = chosen_probs_win
        # mid_dataset['chosen_probs_lose'] = chosen_probs_lose
        # train_dataset = Dataset.from_pandas(mid_dataset)
        # print('=== after add', train_dataset.features)
        # if eval_dataset is not None:
        #     eval_dataset = eval_dataset.map(self.tokenize_row)
        #print('=========')
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset['train'],
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )
        if self.is_deepspeed_enabled:
            self.rm_model = self._prepare_deepspeed(self.rm_model)
            if use_ref:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.rm_model = self.accelerator.prepare_model(self.rm_model, evaluation_mode=True)
            # if use_ref:
            #     self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }
        
        
        filtered_data = self.train_dataset.filter(lambda example:  len(self.tokenizer(example['prompt'])['input_ids'])<1024)
        print(filtered_data)

        # prepare dataloader
        data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))
        return data_loader



    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        """Tokenize a single row from a SPPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt
            prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
            chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
            rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens


        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=batch["rejected_labels"]
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=batch["chosen_labels"]
                )
        #print('batch=======', batch.keys())
        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_rgtg_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            with self.null_ref_context():
                (#all_logps, rgtg_logps, all_logits, rgtg_logits
                    policy_logps,
                    rgtg_logps,
                    ref_logits,
                    _,
                    _,
                    _,
                ) = self.rgtg_forward(self.model,self.rm_model,self.tokenizer, padded_batch)

        return policy_logps, rgtg_logps,ref_logits

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def rgtg_loss(
        self,
        policy_logits: torch.FloatTensor,
        rgtg_logits: torch.FloatTensor,
        ref_logits: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SPPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SPPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        length = policy_logits.shape[1]
        inf_mask = torch.isinf(policy_logits)
        inf_mask_rgtg = torch.isinf(rgtg_logits)
        # print(ref_logits.shape)
        if self.loss_type == "kl":
            losses = (torch.nn.functional.log_softmax(policy_logits,dim=-1,dtype=torch.float32) \
                -torch.nn.functional.log_softmax(rgtg_logits,dim=-1,dtype=torch.float32))\
            * torch.nn.functional.softmax(rgtg_logits,dim=-1,dtype=torch.float32)
            # losses = torch.nn.functional.log_softmax(policy_logits,dim=-1,dtype=torch.float32) \
            # * torch.nn.functional.softmax(rgtg_logits,dim=-1,dtype=torch.float32)
            # print(losses)
            losses=torch.masked_fill(losses, inf_mask, 0)
            # print(losses)
            losses=torch.masked_fill(losses, inf_mask_rgtg, 0)
            # print(losses)
            if self.use_ref:
                ref_losses = (torch.nn.functional.log_softmax(policy_logits,dim=-1,dtype=torch.float32) \
                -torch.nn.functional.log_softmax(ref_logits,dim=-1,dtype=torch.float32))\
                * torch.nn.functional.softmax(ref_logits,dim=-1,dtype=torch.float32)
                # ref_losses = torch.nn.functional.log_softmax(policy_logits,dim=-1,dtype=torch.float32) \
                # * torch.nn.functional.softmax(ref_logits,dim=-1,dtype=torch.float32)
                ref_losses=torch.masked_fill(ref_losses, inf_mask, 0)
                losses = losses + self.beta * ref_losses
            # print(losses)
            losses = torch.sum(losses,dim=-1).view(-1)
            # print(losses)
            losses = -torch.sum(losses,dim=0)/length
            # print(losses)
        elif self.loss_type == "rev_kl":
            losses = torch.nn.functional.log_softmax(rgtg_logits ,dim=-1,dtype=torch.float32) \
            * torch.nn.functional.softmax(policy_logits,dim=-1,dtype=torch.float32)
            losses=torch.masked_fill(losses, inf_mask, 0)
            if self.use_ref:
                ref_losses = torch.nn.functional.log_softmax(ref_logits,dim=-1,dtype=torch.float32) \
                * torch.nn.functional.softmax(policy_logits,dim=-1,dtype=torch.float32)
                ref_losses=torch.masked_fill(ref_losses, inf_mask, 0)
                losses = losses + self.beta * ref_losses            
            losses = torch.sum(losses,dim=-1).view(-1)
            losses = -torch.sum(losses,dim=0)/length

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        return losses

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def partial_kv_cache(self,past_key_values_full,cahce_pos):
        past_key_values_new_list = [[None for i in range(len(past_key_values_full[0]))]for j in range(len(past_key_values_full))]
        for i in range(len(past_key_values_full)):
            for j in range(len(past_key_values_full[i])):
                # past_key_values_new_list[i][j] = past_key_values_full[i][j][:,:,:cahce_pos,:].clone().detach()
                past_key_values_new_list[i][j] = past_key_values_full[i][j][:,:,:cahce_pos,:]
            past_key_values_new_list[i] = tuple(past_key_values_new_list[i])
        past_key_values_new_list = tuple(past_key_values_new_list)
        return past_key_values_new_list

    def duplicate_kv_cache(self,past_key_values_full,n_dups):
        assert len(past_key_values_full[0][0])==1
        past_key_values_new_list = [[None for i in range(len(past_key_values_full[0]))]for j in range(len(past_key_values_full))]
        for i in range(len(past_key_values_full)):
            for j in range(len(past_key_values_full[i])):
                # past_key_values_new_list[i][j] = past_key_values_full[i][j][:,:,:cahce_pos,:].clone().detach()
                past_key_values_new_list[i][j] = past_key_values_full[i][j].repeat(n_dups,1,1,1)
            past_key_values_new_list[i] = tuple(past_key_values_new_list[i])
        past_key_values_new_list = tuple(past_key_values_new_list)
        return past_key_values_new_list

    def count_pos(self,tokens_RM, tokens_all):
        cache_pos = min(tokens_RM['input_ids'].shape[1]-1, tokens_all['input_ids'].shape[1]-1)
        while cache_pos >0:
            if(sum(tokens_RM['input_ids'][:,cache_pos-1] == tokens_all['input_ids'][0,cache_pos-1])==tokens_RM['input_ids'].shape[0]):
                break
            else:
                cache_pos -= 1
        return cache_pos
    
    def create_attention_mask(self,seq_len, bsz=1):
        return torch.ones((bsz, seq_len))
    
    def generate_step(self, mout, input_ids, pre_screen_beam_width=40, weight=0., method="greedy", temperature=0.7, rm_cached=None, debug=True):
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)
        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens.to(expanded_tis.device)))

        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        flat_trme_length = flat_trme.shape[1]
        with torch.no_grad():
            if rm_cached is None:
                rm_out = self.rm_model(**prepare_inputs_for_generation_o(input_ids=flat_trme.to(self.rm_model.device), attention_mask=self.create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm_model.device), past_key_values=None, use_cache=True))
                rm_cached = rm_out.past_key_values
            else:
                rm_out = self.rm_model(**prepare_inputs_for_generation_o(input_ids=flat_trme.to(self.rm_model.device), attention_mask=self.create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm_model.device), past_key_values=rm_cached, use_cache=True))
                del rm_cached
                rm_cached = rm_out.past_key_values
            rewards = rm_out.logits.flatten().to(self.rm_model.device)
            del rm_out, to_rm_eval
            torch.cuda.empty_cache()
            new_scores = rewards * weight + prescreen_logits.flatten()
            policy_score = out_logits.clone().detach()
            rgtg_score = out_logits.clone().detach()
            
            rgtg_score.fill_(-float('inf'))
            # print(f"{rgtg_score.shape=}")
            # print(f"{prescreen_tokens.shape=}")
            # print(f"{new_scores.shape=}")
            rgtg_score[0,prescreen_tokens[0]] = new_scores
            
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
            rm_cached = self.model._reorder_cache(rm_cached, top_k_ids.repeat(pre_screen_beam_width,))

        return flat_trme[top_k_ids.to(flat_trme.device)], rm_cached, rgtg_score, policy_score
        
    def rgtg_forward(
        self, model: nn.Module,rm_model:nn.Module,tokenizer, batch_text: Dict[str, Union[List, str]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # batch = self.tokenizer(batch_text)
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        chat_prompt = tokenizer.apply_chat_template([batch_text['messages'][0][0]],tokenize=False, add_generation_prompt=True)
        chat_inputs = tokenizer.apply_chat_template([batch_text['messages'][0][0]], return_tensors='pt', return_dict=True, add_generation_prompt=True)
        # model.generation_config.pad_token_id = tokenizer.pad_token_id
        # greedy_output = model.generate(**chat_inputs.to(model.device), max_new_tokens=512, num_return_sequences=1, do_sample=False, temperature=1, use_cache=False)
        model.module.generation_config.pad_token_id = tokenizer.pad_token_id
        cur_new_tokens = 0
        last_token_id = None
        cached=None
        rm_cached=None
        tokens = chat_inputs.input_ids.clone().detach()
        rgtg_score_list = []
        policy_score_list = []
        while True:
            # print(cur_new_tokens)
            if cur_new_tokens > 512:
                break
            if tokens[0,-1] == tokenizer.eos_token_id:
                break
            with torch.no_grad():
                if cached is None:
                    mout = self.model(**prepare_inputs_for_generation_o(input_ids=tokens.to(self.model.device), attention_mask=self.create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.model.device), past_key_values=None, use_cache=True))
                    # mout = self.model(input_ids=tokens.to(self.model.device), attention_mask=self.create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.model.device), past_key_values=None, use_cache=True)
                    cached = mout.past_key_values
                    # print(tokens.shape)
                    # print(mout.logits.shape)
                else:
                    
                    mout = self.model(**prepare_inputs_for_generation_o(input_ids=tokens.to(self.model.device), attention_mask=self.create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.model.device), past_key_values=cached, use_cache=True))
                    # tokens_length = tokens.shape[1]               
                    # mout = self.model(input_ids=tokens[:,tokens_length-2:tokens_length-1].to(self.model.device), attention_mask=self.create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.model.device), past_key_values=cached, use_cache=True)
                    # print(tokens.shape)
                    # print(mout.logits.shape)
                    del cached
                    cached = mout.past_key_values
            tokens, rm_cached, cur_rgtg_score,cur_policy_score = self.generate_step(mout, tokens, self.topk, self.rgtg_weight, rm_cached)
            rgtg_score_list.append(cur_rgtg_score)
            policy_score_list.append(cur_policy_score)
            del mout
            torch.cuda.empty_cache()
            cur_new_tokens +=1
        del cached, rm_cached
        torch.cuda.empty_cache()
        greedy_output = tokens
        rgtg_logits= torch.stack(rgtg_score_list,dim=1)
        student_logits = torch.stack(policy_score_list,dim=1)
        # student_logits = torch.stack(policy_score_list,dim=1)
        # self.store_metrics({'generation':tokenizer.decode(greedy_output[0])}, train_eval="train")
        with open(self.args.output_dir + '/greedy_output.jsonl','a') as f:
            json.dump({'greedy_output':tokenizer.decode(greedy_output[0])},f)
            f.write('\n')
        greedy_assistant = tokenizer.decode(greedy_output[0][len(chat_inputs[0]):]).replace(tokenizer.eos_token, "")
        self.store_metrics({'generation_length':float(len(greedy_output[0])-len(chat_inputs[0]))}, train_eval="train")
        all_messages = [batch_text['messages'][0][0], {'role':'assistant', 'content':greedy_assistant}]
        chat_all = tokenizer.apply_chat_template(all_messages,tokenize=False, add_generation_prompt=False)

        batch = self.tokenizer(chat_all, return_tensors='pt' )
        # all_logits = self.model(
        #         **batch.to(self.accelerator.device),
        #         return_dict=True,
        #     ).logits
        all_logits = self.model(
                input_ids=greedy_output.to(self.accelerator.device),
                return_dict=True,
            ).logits
        student_logits = all_logits[:,chat_inputs.input_ids.shape[1]:,:]
        
        if self.use_ref:
            self.ref_model.eval()
            ref_logits = self.ref_model(
                **batch.to(self.accelerator.device),
                return_dict=True,
            ).logits
            ref_logits = ref_logits[:,len(chat_inputs.input_ids):]
            ref_logps = torch.nn.functional.log_softmax(ref_logits,dim=-1,dtype=torch.float32)
        else:
            ref_logits = None
            ref_logps=None
        
        all_logps = torch.nn.functional.log_softmax(student_logits,dim=-1,dtype=torch.float32)

        # rm_output = rm_model(**batch.to(rm_model.device),return_dict=True, use_cache=True)
        rm_output = rm_model(**batch.to(self.accelerator.device),return_dict=True, use_cache=True)
        # print(f"rm_output: {rm_output.logits.item()}")
        metrics = {'rewards': rm_output.logits.item()}
        self.store_metrics(metrics, train_eval="train")

        rgtg_logps = torch.nn.functional.log_softmax(rgtg_logits,dim=-1,dtype=torch.float32)
        return (all_logps, rgtg_logps,ref_logps, student_logits, rgtg_logits, ref_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the SPPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_logps,
            rgtg_logps,
            ref_logps,
            policy_logits,
            rgtg_logits,
            ref_logits
        ) = self.rgtg_forward(model,self.rm_model,self.tokenizer, batch)

        losses= self.rgtg_loss(
            policy_logits,
            rgtg_logits,
            ref_logits
        )

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}logps/policy"] = policy_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/rgtg"] = rgtg_logps.detach().mean().cpu()        
        metrics[f"{prefix}logits/policy"] = policy_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/rgtg"] = rgtg_logits.detach().mean().cpu()
        if self.use_ref:
            metrics[f"{prefix}logps/ref"] = ref_logps.detach().mean().cpu()
            metrics[f"{prefix}logits/ref"] = ref_logits.detach().mean().cpu()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[str, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                reference_output = self.ref_model.generate(
                    input_ids=batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
        
        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "sft" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)