from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    GenerationConfig,
    PreTrainedTokenizer,
)
from trl.trainer.utils import generate


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        p_features = []
        r_queries = []
        for feature in features:
            p_features.append({"input_ids": feature["input_ids"]})
            r_queries.append(feature["r_query_text"])

        p_batch = self.tokenizer.pad(
            p_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": p_batch["input_ids"],
            "r_query_text": r_queries,
        }
        return batch


def get_score(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward logits and the rewards for a given model and query responses.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        # use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )


def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))


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


@torch.no_grad()
def generate_step(mout,lm_backbone,rm_model, input_ids, pre_screen_beam_width=40, weight=0., method="greedy", temperature=0.7, rm_cached=None, debug=True):
    out_logits = mout.logits[:, -1]
    prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)
    expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)

    to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens.to(expanded_tis.device)))

    flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
    flat_trme_length = flat_trme.shape[1]
    with torch.no_grad():
        if rm_cached is None:
            rm_out = rm_model(**prepare_inputs_for_generation_o(input_ids=flat_trme.to(rm_model.device), attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(rm_model.device), past_key_values=None, use_cache=True))
            rm_cached = rm_out.past_key_values
        else:
            rm_out = rm_model(**prepare_inputs_for_generation_o(input_ids=flat_trme.to(rm_model.device), attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(rm_model.device), past_key_values=rm_cached, use_cache=True))
            del rm_cached
            rm_cached = rm_out.past_key_values
        rewards = rm_out.logits.flatten().to(rm_model.device)
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
        rm_cached = lm_backbone._reorder_cache(rm_cached, top_k_ids.repeat(pre_screen_beam_width,))

    return flat_trme[top_k_ids.to(flat_trme.device)], rm_cached, rgtg_score, policy_score


@torch.no_grad()
def generate_rgtg(
    lm_backbone: torch.nn.Module,rm_model: torch.nn.Module,tokenizer, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    cur_new_tokens = 0
    cached=None
    rm_cached=None
    tokens = input_ids.clone().detach()
    rgtg_score_list = []
    policy_score_list = []
    topk=20
    rgtg_weight=0.5
    while True:
        if cur_new_tokens > 512:
            break
        if tokens[0,-1] == tokenizer.eos_token_id:
            break
        with torch.no_grad():
            if cached is None:
                mout = lm_backbone(**prepare_inputs_for_generation_o(input_ids=tokens.to(lm_backbone.device), attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(lm_backbone.device), past_key_values=None, use_cache=True))
                cached = mout.past_key_values
            else:
                mout = lm_backbone(**prepare_inputs_for_generation_o(input_ids=tokens.to(lm_backbone.device), attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(lm_backbone.device), past_key_values=cached, use_cache=True))
                del cached
                cached = mout.past_key_values
        tokens, rm_cached, cur_rgtg_score,cur_policy_score = generate_step(mout,lm_backbone,rm_model, tokens, topk, rgtg_weight, rm_cached)
        rgtg_score_list.append(cur_rgtg_score)
        policy_score_list.append(cur_policy_score)
        torch.cuda.empty_cache()
        cur_new_tokens +=1
    del cached, rm_cached
    torch.cuda.empty_cache()
    greedy_output = tokens
    rgtg_logits= torch.stack(rgtg_score_list,dim=1)
    student_logits = torch.stack(policy_score_list,dim=1)
    logits = rgtg_logits
    return tokens, logits


@torch.no_grad()
def batch_generation_rgtg(
    model: torch.nn.Module,
    rm_model: torch.nn.Module,
    tokenizer,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
):
    query_responses = []
    logitss = []
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate_rgtg(
            model,
            rm_model,
            tokenizer,
            query,
            pad_token_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)
        max_query_response_len = max(query_response.shape[1] for query_response in query_responses)
        max_sequence_len = max(logit.shape[1] for logit in logitss)
        query_responses = [
            torch.nn.functional.pad(query_response, (0, max_query_response_len - query_response.shape[1]), value=pad_token_id)
            for query_response in query_responses
        ]
        logitss = [
            torch.nn.functional.pad(logits, (0, 0, 0, max_sequence_len - logits.shape[1]), value=pad_token_id)
            for logits in logitss
        ]
    return torch.cat(query_responses, 0), torch.cat(logitss, 0)


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
):
    query_responses = []
    logitss = []
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)
    max_query_response_len = max(query_response.shape[1] for query_response in query_responses)
    max_sequence_len = max(logit.shape[1] for logit in logitss)
    query_responses = [
        torch.nn.functional.pad(query_response, (0, max_query_response_len - query_response.shape[1]), value=pad_token_id)
        for query_response in query_responses
    ]
    logitss = [
        torch.nn.functional.pad(logits, (0, 0, 0, max_sequence_len - logits.shape[1]), value=pad_token_id)
        for logits in logitss
    ]
    return torch.cat(query_responses, 0), torch.cat(logitss, 0)
