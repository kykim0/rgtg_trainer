from datasets import load_dataset
import argparse
import gc
import json
from pathlib import Path
from tqdm import tqdm
import time
import transformers
from tqdm import tqdm
import torch
import wandb 
import json
import torch.nn.functional as F
import warnings
from transformers.utils import logging
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")
logging.get_logger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
parser.add_argument("--split", type=str, default="train_prefs")
parser.add_argument("--run_percent", type=float, default=100.)
parser.add_argument("--rm", type=str)
parser.add_argument("--llm", type=str)
parser.add_argument("--max_new_token", type=int, default=512)
parser.add_argument("--window_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-05)
parser.add_argument("--weight", type=float, default=1.)
parser.add_argument("--on_policy", action="store_true")
parser.add_argument("--uncertainty", action="store_true")
parser.add_argument("--chosen_test", action="store_true")
parser.add_argument("--uncertainty_threshold", type=float, default=1.)
parser.add_argument("--length_normalization", action="store_true")
parser.add_argument("--exp_reward_norm", action="store_true")
parser.add_argument("--maxsteps", type=int, default=-1)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--llm_gpu", type=str, default="cuda:0")
parser.add_argument("--rm_gpu", type=str, default="cuda:1")
parser.add_argument("--save_path", type=str, default="./model_checkpoint")  # Path to save model
parser.add_argument("--log_file", type=str, default="./generation_results.jsonl")  # Path to save log and results
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--lookup_step", type=int, default=10)
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--reverse", action="store_true")



args = parser.parse_args()

print(f"{args=}")

topk=args.topk

# Initialize wandb project
wandb.init(project="RM Guided Self Knowledge Distillation", config=args)
wandb.config.update(args)  # Update wandb config with parsed arguments


print(f"[INFO]: Loading dataset ({args.dataset=}, {args.split=})")
test_ds = load_dataset(args.dataset, split=args.split)

    
end_idx = int(len(test_ds) * (args.run_percent/100.))
print(f"[INFO]: {end_idx=}, {len(test_ds)=}")
# test_ds = test_ds[:end_idx]

# for idx, ds_row in enumerate(tqdm(truncated_ds)):
#     current_prompt = ds_row

def entropy(logits):
    prob = F.softmax(logits, dim=-1)
    log_prob = torch.log_softmax(logits, dim=-1)
    return -torch.sum(prob * log_prob, dim=-1)
    
def partial_kv_cache(past_key_values_full,cahce_pos):
    past_key_values_new_list = [[None for i in range(len(past_key_values_full[0]))]for j in range(len(past_key_values_full))]
    for i in range(len(past_key_values_full)):
        for j in range(len(past_key_values_full[i])):
            # past_key_values_new_list[i][j] = past_key_values_full[i][j][:,:,:cahce_pos,:].clone().detach()
            past_key_values_new_list[i][j] = past_key_values_full[i][j][:,:,:cahce_pos,:]
        past_key_values_new_list[i] = tuple(past_key_values_new_list[i])
    past_key_values_new_list = tuple(past_key_values_new_list)
    return past_key_values_new_list

def duplicate_kv_cache(past_key_values_full,n_dups):
    assert len(past_key_values_full[0][0])==1
    past_key_values_new_list = [[None for i in range(len(past_key_values_full[0]))]for j in range(len(past_key_values_full))]
    for i in range(len(past_key_values_full)):
        for j in range(len(past_key_values_full[i])):
            # past_key_values_new_list[i][j] = past_key_values_full[i][j][:,:,:cahce_pos,:].clone().detach()
            past_key_values_new_list[i][j] = past_key_values_full[i][j].repeat(n_dups,1,1,1)
        past_key_values_new_list[i] = tuple(past_key_values_new_list[i])
    past_key_values_new_list = tuple(past_key_values_new_list)
    return past_key_values_new_list

def count_pos(tokens_RM, tokens_all):
    cache_pos = min(tokens_RM['input_ids'].shape[1]-1, tokens_all['input_ids'].shape[1]-1)
    while cache_pos >0:
        if(sum(tokens_RM['input_ids'][:,cache_pos-1] == tokens_all['input_ids'][0,cache_pos-1])==tokens_RM['input_ids'].shape[0]):
            break
        else:
            cache_pos -= 1
    return cache_pos

model = transformers.AutoModelForCausalLM.from_pretrained(args.llm, device_map=args.llm_gpu,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2')
tokenizer = transformers.AutoTokenizer.from_pretrained(args.llm)
# model.train()
if not args.on_policy:
    model_ref = transformers.AutoModelForCausalLM.from_pretrained(args.llm, device_map=args.rm_gpu,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2')
    model_ref.eval()
# tokenizer = transformers.AutoTokenizer.from_pretrained(args.llm)
if 'gemma' in args.llm:
    tokenizer.eos_token = '<|im_end|>'
    tokenizer.eos_token_id = 107
    
rm_model = transformers.AutoModelForSequenceClassification.from_pretrained(args.rm, device_map=args.rm_gpu,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2')
rm_tokenizer = transformers.AutoTokenizer.from_pretrained(args.rm)
rm_model.eval()
# if rm_tokenizer.pad_token is None:
#     rm_model.config.pad_token_id = rm_model.config.eos_token_id
#     rm_tokenizer.pad_token=rm_tokenizer.eos_token
#     rm_tokenizer.pad_token_id=rm_tokenizer.eos_token_id

def get_optimizer_params(model: torch.nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters

param_groups = get_optimizer_params(model)
optimizer = transformers.AdamW(param_groups, args.lr)
look_size=args.lookup_step
for i in tqdm(range(len(test_ds))):
    if i<args.start_index:
        continue
    if args.maxsteps>0 and i>args.maxsteps:
        break
    current_prompt = test_ds[i]
    if len(current_prompt['messages'][0]['content'])>1024:
        continue
    if 'pythia' in args.llm:
        chat_prompt = tokenizer.apply_chat_template([{'role':'system','content':''},current_prompt['messages'][0]],tokenize=False, add_generation_prompt=True)
        chat_inputs = tokenizer.apply_chat_template([{'role':'system','content':''},current_prompt['messages'][0]], return_tensors='pt', return_dict=True, add_generation_prompt=True)
        print(f"{chat_inputs.input_ids.shape[1]=}")
        if chat_inputs.input_ids.shape[1]>1024:
            print("Too long to LLM")
            continue
    else:
        if tokenizer.bos_token:
            chat_prompt = tokenizer.apply_chat_template([current_prompt['messages'][0]],tokenize=False, add_generation_prompt=True).replace(tokenizer.bos_token,'')
        else:
            chat_prompt = tokenizer.apply_chat_template([current_prompt['messages'][0]],tokenize=False, add_generation_prompt=True)

    with torch.no_grad():
        if args.chosen_test:
            greedy_assistant = current_prompt['messages'][1]['content']
        elif args.on_policy:
            # context_length = chat_inputs.shape[1]
            # [context_length:]
            greedy_output = model.generate(**chat_inputs.to(model.device), max_new_tokens=512, num_return_sequences=1, do_sample=False, temperature=1, use_cache=False)
            greedy_assistant = tokenizer.decode(greedy_output[0][len(chat_inputs[0]):]).replace(tokenizer.eos_token, "")
        else:
            greedy_output = model_ref.generate(**chat_inputs.to(model_ref.device), max_new_tokens=512, num_return_sequences=1, do_sample=False, temperature=1,  eos_token_id=tokenizer.eos_token_id, use_cache=False)
            greedy_assistant = tokenizer.decode(greedy_output[0][len(chat_inputs[0]):]).replace(tokenizer.eos_token, "")

    if 'pythia' in args.llm:
        all_messages = [{"role": "system", "content": ""}, current_prompt['messages'][0], {'role':'assistant', 'content':greedy_assistant}]
    else:
        all_messages = [current_prompt['messages'][0], {'role':'assistant', 'content':greedy_assistant}]


    # if 'pythia' in args.llm:
    #     chat_all = tokenizer.decode(greedy_output[0])
    #     chat_all_inputs = {'input_ids':greedy_output}
    # else:
    chat_all = tokenizer.apply_chat_template(all_messages,tokenize=False, add_generation_prompt=False)
    chat_all_inputs = tokenizer(chat_all, return_tensors='pt').to(model.device)#greedy_output.sequences[0])

    
    all_outputs = model(**chat_all_inputs, return_dict=True,use_cache=True)
    if look_size >1:
        past_key_values_llm = duplicate_kv_cache(all_outputs.past_key_values,topk)

    out_logits = all_outputs.logits
    if args.uncertainty:
        out_uncertainty = entropy(out_logits)
    prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=topk)
    if 'pythia' in args.llm:
        chat_prompt = tokenizer.apply_chat_template([{"role": "system", "content": ""}, current_prompt['messages'][0]],tokenize=False, add_generation_prompt=True)
    else:
        chat_prompt = tokenizer.apply_chat_template([current_prompt['messages'][0]],tokenize=False, add_generation_prompt=True)
    chat_inputs = tokenizer(chat_prompt, return_tensors='pt').to(model.device)
    start_response = len(chat_inputs.input_ids[0])
    end_response = prescreen_logits.shape[1]
    student_logits = out_logits[:,range(end_response-1,start_response-1,-args.window_size)]
    guided_logits = student_logits.clone().detach().to(student_logits.device)

    start = time.time()
    # del chat_inputs
    # torch.cuda.empty_cache()

    all_rm_chat = rm_tokenizer.apply_chat_template(all_messages,tokenize=False, add_generation_prompt=False)
    all_rm_inputs = rm_tokenizer.apply_chat_template(all_messages,return_tensors='pt',return_dict=True, add_generation_prompt=False).to(rm_model.device)
    
    if all_rm_inputs['input_ids'].shape[1]>2048:
        print("Too long to RM")
        del (guided_logits,all_outputs,greedy_output)
        del (all_rm_inputs)
        if look_size >1:
            del past_key_values_llm
        # del past_key_values_llm_candidates
        torch.cuda.empty_cache()
        
        continue
    with torch.no_grad():
        rm_model_outputs_all = rm_model(**all_rm_inputs, return_dict=True, use_cache=True)
    past_key_values_all = duplicate_kv_cache(rm_model_outputs_all.past_key_values,topk)
        
    gen_time = 0
    rm_time = 0
    ## Get Guided Logits
    cnt_target = 0
    for idx in range(end_response-1,start_response-1,-args.window_size):
        
        if args.uncertainty and out_uncertainty[0,idx-1]<args.uncertainty_threshold:
            continue
        cnt_target+=1
    # for idx in range(start_response,end_response):
        gen_start = time.time()
        cur_response_ids = prescreen_tokens[0:1,idx-1,:]
        llm_inputs_cadidates = torch.concat((chat_all_inputs.input_ids[:,:idx].repeat(topk,1),cur_response_ids.T),dim=1)
        
        if look_size <=1:
            cur_response=tokenizer.batch_decode(llm_inputs_cadidates[:,chat_inputs.input_ids.shape[1]:])
            cur_rm_inputs = torch.concat((llm_inputs_cadidates,torch.IntTensor([[tokenizer.eos_token_id]]).to(llm_inputs_cadidates.device).repeat(topk,1)),dim=1)
            del llm_inputs_cadidates
            torch.cuda.empty_cache()
            # print('###',idx, cur_response)
            # cur_messages = [[current_prompt['messages'][0], {'role':'assistant', 'content':cur_response[turnk]}] for turnk in range(len(cur_response))]
            # cur_rm_chat = rm_tokenizer.apply_chat_template(cur_messages,tokenize=False, add_generation_prompt=False)
            # cur_rm_inputs = rm_tokenizer(cur_rm_chat, return_tensors='pt',padding=True).to(rm_model.device)
        
        elif look_size >1:
            past_key_values_llm_candidates = partial_kv_cache(past_key_values_llm,idx)
            with torch.no_grad():
                x_generated = model.generate(input_ids = llm_inputs_cadidates, past_key_values=past_key_values_llm_candidates, max_new_tokens=look_size-1, do_sample=False, eos_token_id=tokenizer.eos_token_id, use_cache=True)
            cur_response=tokenizer.batch_decode(x_generated)
            # print(cur_response)
            del x_generated, llm_inputs_cadidates
            torch.cuda.empty_cache()
            cur_messages = [[current_prompt['messages'][0], {'role':'assistant', 'content':cur_response[turnk]}] for turnk in range(len(cur_response))]
            cur_rm_chat = rm_tokenizer.apply_chat_template(cur_messages,tokenize=False, add_generation_prompt=False)
            cur_rm_inputs = rm_tokenizer(cur_rm_chat, return_tensors='pt',padding=True).to(rm_model.device)
        
        ##get current_cache_pos by counting the number of same prefixed tokens
        if 'pythia' in args.llm and look_size == 1:
            # print(cur_rm_inputs)
            cache_pos = cur_rm_inputs.shape[1]-2
            ##get partial_kv_cache
            past_key_values_partial = partial_kv_cache(past_key_values_all,cache_pos)
            partial_rm_input_ids = cur_rm_inputs[:,cache_pos:]
        else:
            cache_pos = count_pos(cur_rm_inputs,all_rm_inputs)
            ##get partial_kv_cache
            past_key_values_partial = partial_kv_cache(past_key_values_all,cache_pos)
            partial_rm_input_ids = cur_rm_inputs['input_ids'][:,cache_pos:]
        
        with torch.no_grad():
            # rm_model_outputs = rm_model(**cur_rm_inputs, return_dict=True)
            if 'pythia' in args.llm and look_size == 1:
                rm_model_outputs = rm_model(input_ids = partial_rm_input_ids,past_key_values = past_key_values_partial,use_cache=True, return_dict=True)

            else:
                rm_model_outputs = rm_model(input_ids = partial_rm_input_ids,attention_mask=cur_rm_inputs['attention_mask'],past_key_values = past_key_values_partial,use_cache=True, return_dict=True)
        # time_rm = time.time()-rm_start
        # rm_time +=time_rm
        # print(f"####idx:{idx}, {time_rm=}, {rm_time=}")
        rm_scores = rm_model_outputs.logits
        w=args.weight
        if args.exp_reward_norm:
            cur_guided_score = w * rm_scores[:,0].to(prescreen_logits.device)
        else:
            cur_guided_score = prescreen_logits[0,idx-1,:] + w * rm_scores[:,0].to(prescreen_logits.device)
        # guided_logits[:,idx-start_response,:]=-float('inf')
        # guided_logits[:,idx-start_response,prescreen_tokens[0,idx-1,:]] = cur_guided_score.to(guided_logits.device)
        if look_size >1:
            del past_key_values_llm_candidates
        
        guided_logits[:,guided_logits.shape[1]-cnt_target,prescreen_tokens[0,idx-1,:]] = cur_guided_score.to(guided_logits.device)
        del rm_model_outputs, cur_guided_score
        torch.cuda.empty_cache()
    gc.collect()
    ## Calculate Distillation Loss
    if args.reverse:
        guided_log_probs = torch.nn.functional.log_softmax(guided_logits,dim=-1,dtype=torch.float32)
        inf_mask = torch.isinf(guided_logits)

        student_probs = torch.nn.functional.softmax(student_logits,dim=-1,dtype=torch.float32)
        # inf_mask = torch.isinf(student_logits)
        prod_probs = student_probs * guided_log_probs
        prod_probs=torch.masked_fill(prod_probs, inf_mask, 0)

    else:
        if args.length_normalization:
            response_range = torch.arange(end_response-start_response, 0, -1,dtype=torch.float32).to(guided_logits.device)
            length_norm_mask = 1.0 / response_range.unsqueeze(-1)
            length_norm_mask = length_norm_mask.expand(1, end_response - start_response, all_outputs.logits.shape[2]).to(guided_logits.device)
            guided_probs = torch.nn.functional.softmax(guided_logits,dim=-1,dtype=torch.float32)
            guided_probs = guided_probs * length_norm_mask
        else:
            guided_probs = torch.nn.functional.softmax(guided_logits,dim=-1,dtype=torch.float32)
        
        if args.exp_reward_norm:
            guided_probs = guided_probs * torch.nn.functional.softmax(student_logits,dim=-1,dtype=torch.float32)
        log_probs = torch.nn.functional.log_softmax(student_logits,dim=-1,dtype=torch.float32)
        # inf_mask = torch.isinf(guided_logits)
        prod_probs = guided_probs *log_probs
    # prod_probs=torch.masked_fill(prod_probs, inf_mask, 0)

    # loss_func(student_logits.view(-1,student_logits.shape[-1]),guided_logits.view(-1,guided_logits.shape[-1]))
    x = torch.sum(prod_probs,dim=-1).view(-1)
    
    # distill_loss = -torch.sum(x,dim=0)/prod_probs.shape[0]
    if args.window_size>1 or args.uncertainty:
        distill_loss = -torch.sum(x,dim=0)/cnt_target
    elif args.length_normalization:
        distill_loss = -torch.sum(x,dim=0)
    else:
        distill_loss = -torch.sum(x,dim=0)/x.shape[0]
    distill_loss = distill_loss / args.gradient_accumulation_steps
    distill_loss.backward()
    
    wandb.log({"Distill Loss": distill_loss.item(), "Reward Score": rm_model_outputs_all.logits[0].item()})
    with open(args.log_file, "a") as log_file:
        json.dump({"Reward Score": rm_model_outputs_all.logits[0].item(),"Count Target":cnt_target,"Prompt": current_prompt['messages'][0]['content'], "Greedy Assistant": greedy_assistant}, log_file)
        log_file.write("\n")
    if i % 1 ==0:
        # for p in model.parameters():
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # Log the distillation loss and reward score to wandb
        print(f"###Iteration{i}/{len(test_ds)}### Distill Loss: {distill_loss}, Reward Score: {rm_model_outputs_all.logits[0].item()}")
    del (cur_rm_inputs,all_rm_inputs,rm_scores)    
    del (chat_all_inputs,all_outputs,guided_logits)
    torch.cuda.empty_cache()
    
    if (i + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()

        # Save Hugging Face model checkpoint periodically (every 1000 iterations)
    if (i + 1) % args.save_steps == 0:
        model.save_pretrained(f"{args.save_path}/checkpoint_{i+1}")
        tokenizer.save_pretrained(f"{args.save_path}/checkpoint_{i+1}")
        print(f"Hugging Face model checkpoint saved at iteration {i+1}")


    
# Final model save after all iterations
model.save_pretrained(f"{args.save_path}/final_model")
tokenizer.save_pretrained(f"{args.save_path}/final_model")
print("Final Hugging Face model saved.")
    # model.step()
    
# Finalize the wandb run
wandb.finish()