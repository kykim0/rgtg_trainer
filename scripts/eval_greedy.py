# Example usage:
# $ CUDA_VISIBLE_DEVICES=0 python3 etc/eval_greedy.py \
#     --llm ${ckpt_dir} --rm ${rmodel} --ref_llm ${ref_model} \
#     --log_file ${eval_dir}/${ckpt_base}.jsonl \
#     --batch_size 4 --max_new_tokens 512 --split test_prefs

import argparse
import json
import os

from datasets import load_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
)
from trl.trainer.utils import (
    first_true_indices,
    forward,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="kykim0/ultrafeedback_binarized_cleaned_20p")
parser.add_argument("--split", type=str, default="test_prefs")
parser.add_argument("--num_samples", type=int, default=None)
parser.add_argument("--rm", type=str, default="kykim0/pythia-1b-tulu-v2-mix-uf-rm")
parser.add_argument("--llm", type=str, default="kykim0/pythia-1b-tulu-v2-mix")
parser.add_argument("--ref_llm", type=str, default=None)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--llm_gpu", type=str, default="cuda:0")
parser.add_argument("--rm_gpu", type=str, default="cuda:0")
parser.add_argument('--approx_kld', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--log_file", type=str, default="./generation_results.jsonl")  # Path to save log and results

args = parser.parse_args()


def kld(logits, ref_logits, responses, sequence_lengths):
    """Computes the KL divergence between the distribs for the given logits.

    In case `response` is given, compute the approximate KLD, the one often
    used in PPO training (logprobs - ref_logprobs).
    """
    # Correctly applying padding mask is crucial in KL computation.
    batch_size = responses.shape[0]
    response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(batch_size, 1)
    padding_mask = response_idxs > sequence_lengths.unsqueeze(1)

    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    ref_logprobs = F.log_softmax(ref_logits, dim=-1, dtype=torch.float32)
    if args.approx_kld:
        logprobs = torch.gather(logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
        ref_logprobs = torch.gather(ref_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
        ratios = torch.masked_fill(logprobs - ref_logprobs, padding_mask, 0)
        klds = ratios.sum(1).detach().cpu().numpy()
        assert(klds.shape) == (batch_size,)
        return klds

    probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # [b, l, v]
    inf_mask = torch.isinf(ref_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(probs * logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(probs * ref_logprobs, inf_mask, 0)
    ratios = torch.masked_fill(torch.sum(prod_probs, dim=-1), padding_mask, 0)
    klds = ratios.sum(1).detach().cpu().numpy()
    assert(klds.shape) == (batch_size,)
    return klds


def main():
    if os.path.exists(args.log_file):
        print(f"Eval log file exists: {args.log_file}")
        return

    model = AutoModelForCausalLM.from_pretrained(args.llm, device_map=args.llm_gpu, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    tokenizer.padding_side = "left"
    model.eval()

    if args.ref_llm:
        ref_model = AutoModelForCausalLM.from_pretrained(args.ref_llm, device_map=args.llm_gpu, torch_dtype=torch.bfloat16)
        ref_model.eval()

    rmodel = AutoModelForSequenceClassification.from_pretrained(args.rm, device_map=args.rm_gpu, torch_dtype=torch.bfloat16)
    rtokenizer = AutoTokenizer.from_pretrained(args.rm)
    rmodel.eval()

    dataset = load_dataset(args.dataset, split=args.split)
    if args.num_samples:
        dataset = dataset.select(range(args.num_samples))

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    batch_size = args.batch_size
    output_to_write = []
    for idx in tqdm(range(0, len(dataset), batch_size)):
        query_texts = []
        examples = dataset[idx:(idx + batch_size)]
        for prompt in examples["prompt"]:
            messages = [{"role": "user", "content": prompt}]
            query_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(query_texts, return_tensors="pt", padding=True).to(model.device)
        context_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_dict = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            outputs = output_dict.sequences
            seq_lengths = first_true_indices(outputs[:, context_length:] == tokenizer.pad_token_id) - 1
            output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            output_texts = [output_text.replace(tokenizer.pad_token, "") for output_text in output_texts]

            rm_texts = [q + r for q, r in zip(query_texts, output_texts)]
            rm_inputs = rtokenizer(rm_texts, return_tensors="pt", padding=True).to(rmodel.device)
            scores = rmodel(**rm_inputs).logits.detach().cpu()
            scores = [score.squeeze().to(dtype=torch.float).item() for score in scores]

        if ref_model:
            logits = torch.stack(output_dict.scores, 1)
            ref_output = forward(ref_model, outputs, tokenizer.pad_token_id)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            responses = outputs[:, context_length:]
            klds = kld(logits, ref_logits, responses, seq_lengths)

        seq_lengths = seq_lengths.detach().cpu().numpy()
        for idx, (prompt, chosen, rejected, output_text, sequence_length, score) in enumerate(zip(
            examples["prompt"], examples["chosen"], examples["rejected"], output_texts, seq_lengths, scores
        )):
            response_idx = output_text.find("<|assistant|>\n") + len("<|assistant|>\n")
            output_text = output_text[response_idx:].replace(tokenizer.pad_token, "")
            out_dict = {
                "prompt": prompt,
                # "chosen": chosen[-1]["content"],
                # "rejected": rejected[-1]["content"],
                "generated": output_text,
                "length": int(sequence_length),
                "score": round(float(score), 5),
            }
            if ref_model:
                out_dict.update({
                    "kld": round(float(klds[idx]), 5),
                })
            output_to_write.append(out_dict)

    with open(args.log_file, "w") as f:
        for out_to_write in output_to_write:
            f.write(json.dumps(out_to_write) + "\n")


if __name__ == "__main__":
    main()
