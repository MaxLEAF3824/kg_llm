from click import group
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tokenizers import Tokenizer
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import random
from tqdm.auto import tqdm
import pickle
from functools import partial
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count, Manager
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import uuid
import bisect
import csv

class EntityDataset(Dataset):
    def __init__(self, data_path: str, kg_path : str, tokenizer: Tokenizer, size=None, max_len=9999999, add_dash=True, dash_token='[DASH]', *args, **kwargs):
        self.max_len = max_len
        self.add_dash = add_dash
        self.dash_token = dash_token
        self.tokenizer = tokenizer
        self.prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n##USER:\n{input}\n\n##ASSISTANT:\n{output} " + self.tokenizer.eos_token
        self.begin_out_ids = self.tokenizer("\n\n##ASSISTANT:\n", add_special_tokens=False)['input_ids'][1:]
        self.kg = [{'source':s, 'target':t, 'edge':e} for (s,t,e) in csv.reader(open(kg_path, 'r').readlines()[1:])]
        self.data = json.load(open(data_path))
        self.data = self.data[:size] if size else self.data
        self.length = len(self.data)
        print(f"Loaded dataset with {len(self)} elements")
    
    
    def make_inputs(self):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        hard_position_type_ids_list = []
        position_ids_list = []

        # 多线程        
        import concurrent.futures
        thread_num = 2
        make_input_func = partial(self.make_input_func, kg=self.kg, tok=self.tokenizer)
        with concurrent.futures.ThreadPoolExecutor(thread_num) as executor:
            future_to_input = {executor.submit(make_input_func, ins): ins for ins in self.data}
            for future in tqdm(concurrent.futures.as_completed(future_to_input),total=len(self.data)):
                ins = future_to_input[future]
                try:
                    output = future.result()
                    input_ids_list.append(output[0])
                    attention_mask_list.append(output[1])
                    labels_list.append(output[2])
                    hard_position_type_ids_list.append(output[3])
                    position_ids_list.append(output[4])
                except Exception as exc:
                    print(f'An exception occurred: {exc}')
        
        # 单进程
        # make_input_func = partial(self.make_input_func, kg=self.kg, tok=self.tokenizer)
        # for ins in tqdm(self.data, total=len(self.data)):
        #     output = make_input_func(ins)
        #     input_ids_list.append(output[0])
        #     attention_mask_list.append(output[1])
        #     labels_list.append(output[2])
        #     hard_position_type_ids_list.append(output[3])
        #     position_ids_list.append(output[4])
        
        return input_ids_list, attention_mask_list, labels_list, hard_position_type_ids_list, position_ids_list
    
    def make_input_func(self, ins, kg, tok):
        all_text = self.prompt_template.format(input=ins['input'], output=ins['output']).strip()
        inp = tok(all_text)
        input_ids = inp['input_ids']
        et = list({e:t for e,t in zip(ins['input_entities']+ins['output_entities'], ins['input_triplets']+ins['output_triplets']) if len(t) > 0}.items())
        et = sorted(et, key=lambda et: -len(tok(et[0], add_special_tokens=False)['input_ids'])) # 按entity的长度从长到短排序
        # print('et: ', et)
        
        # 识别input_ids中的entity并替换为0,-1,-2...
        for i,(e,t) in enumerate(et):
            e_ids = tok(e, add_special_tokens=False)['input_ids']
            window_size = len(e_ids)
            new_input_ids = []
            idx = 0
            while idx < len(input_ids):
                if idx+window_size<len(input_ids):
                    if input_ids[idx:idx + window_size] == e_ids:
                        new_input_ids.append(-i)
                        idx += window_size
                        continue
                new_input_ids.append(input_ids[idx])
                idx += 1
            input_ids = new_input_ids
        
        has_entity = min(input_ids) < 0

        # 拓展input_ids并计算每个token的hard_position_type_id和group_id, 0代表non-entity tokens，1代表entity tokens, 2代表triplet tokens, 3代表triplet target tokens
        if has_entity:
            hard_position_type_ids = []
            group_ids = []
            new_input_ids = []
            idx = 0
            group_idx = 0
            while idx < len(input_ids):
                if input_ids[idx] <= 0:
                    e, t = et[-input_ids[idx]]
                    # print('t: ', t)
                    e_ids = tok(e, add_special_tokens=False)['input_ids']
                    new_input_ids.extend(e_ids)
                    hard_position_type_ids.extend([1] * len(e_ids))
                    group_ids.extend([-1] * len(e_ids))
                    sample_num = 10
                    tids = random.sample(t, min(sample_num, len(t)))
                    triplets = [kg[tid] for tid in tids]
                    for triplet in triplets:
                        tri_prompt = f"[DASH]" if self.add_dash else ""
                        tri_prompt_id = tok(tri_prompt, add_special_tokens=False)['input_ids']
                        
                        # tri_source_id = tok(triplet['source'], add_special_tokens=False)['input_ids']
                        tri_edge_id = tok(triplet['edge'], add_special_tokens=False)['input_ids']
                        tri_target_id = tok(triplet['target'] + " " +tok.eos_token, add_special_tokens=False)['input_ids']
                        
                        new_input_ids.extend(tri_prompt_id)
                        # new_input_ids.extend(tri_source_id)
                        new_input_ids.extend(tri_edge_id)
                        new_input_ids.extend(tri_target_id)
                        
                        hard_position_type_ids.extend([2] * len(tri_prompt_id))
                        # hard_position_type_ids.extend([3] * len(tri_source_id))
                        hard_position_type_ids.extend([2] * len(tri_edge_id))
                        hard_position_type_ids.extend([3] * len(tri_target_id))
                        
                        group_ids.extend([group_idx] * len(tri_prompt_id))
                        # group_ids.extend([group_idx] * len(tri_source_id))
                        group_ids.extend([group_idx] * len(tri_edge_id))
                        group_ids.extend([group_idx] * len(tri_target_id))
                        group_idx += 1
                    idx += 1
                else:
                    new_input_ids.append(input_ids[idx])
                    hard_position_type_ids.append(0) # 0 means original text
                    group_ids.append(-1) # -1 means original text
                    idx += 1
        else:
            hard_position_type_ids = [0] * len(input_ids)
            group_ids = []
            new_input_ids = input_ids
        
        input_ids = new_input_ids
        seq_len = len(input_ids)

        # 利用hard_position_type_ids计算attention_mask_map, shape为seq_len*seq_len 0代表non-entity tokens，1代表entity tokens, 2代表triplet tokens, 3代表triplet target tokens
        attention_mask = torch.ones(seq_len,seq_len, dtype=torch.bool)
        attention_mask = torch.tril(attention_mask, diagonal=0)
        position_ids = torch.arange(seq_len, dtype=torch.long)
        hard_position_type_ids = torch.tensor(hard_position_type_ids)
        
        if has_entity:
            original_index = (hard_position_type_ids <= 1).nonzero().view(-1)
            original_index_index = (original_index[1:] - original_index[:-1] - 1).nonzero().view(-1)
            original_index_start_end = original_index[torch.concat([original_index_index + 1,original_index_index]).tolist() + [0, -1]].sort()[0]
            original_index_range = [(original_index_start_end[2*i].item(),original_index_start_end[2*i+1].item()+1) for i in range(len(original_index_start_end)//2)]
            # print('original_index_range: ', original_index_range)
            entity_index = (hard_position_type_ids == 1).nonzero().view(-1)
            entity_index_index = (entity_index[1:] - entity_index[:-1] - 1).nonzero().view(-1)
            entity_index_start_end = entity_index[torch.concat([entity_index_index + 1,entity_index_index]).tolist() + [0, -1]].sort()[0]
            entity_index_range = [(entity_index_start_end[2*i].item(),entity_index_start_end[2*i+1].item()+1) for i in range(len(entity_index_start_end)//2)]
            # print('entity_index_range: ', entity_index_range)
            triplet_index = (hard_position_type_ids >= 2).nonzero().view(-1)
            triplet_index_index = (triplet_index[1:] - triplet_index[:-1] - 1).nonzero().view(-1)
            triplet_index_start_end = triplet_index[torch.concat([triplet_index_index + 1,triplet_index_index]).tolist() + [0, -1]].sort()[0]
            triplet_index_range = [(triplet_index_start_end[2*i].item(),triplet_index_start_end[2*i+1].item()+1) for i in range(len(triplet_index_start_end)//2)]
            # print('triplet_index_range: ', triplet_index_range)
            position_ids[hard_position_type_ids <= 1] = torch.arange(len(original_index))

            # 每个original token看不到的triplet group
            for oir in original_index_range:
                for tir in triplet_index_range:
                    attention_mask[oir[0]:oir[1],tir[0]:tir[1]] = False
            # 每个triplet group只看得到自己前面一个entity
            for i, tir in enumerate(triplet_index_range):
                attention_mask[tir[0]:tir[1],:tir[0]] = False
                cur_eir = entity_index_range[bisect.bisect_right([x[1] for x in entity_index_range], tir[0])-1]
                attention_mask[tir[0]:tir[1],cur_eir[0]:cur_eir[1]] = True
                position_ids[tir[0]:tir[1]] = torch.arange(tir[1]-tir[0])+1+position_ids[tir[0]-1]
        
        output_start_idx = None
        for i in range(len(input_ids)):
            if input_ids[i:i+len(self.begin_out_ids)] == self.begin_out_ids:
                output_start_idx = i + len(self.begin_out_ids)
                break
        assert output_start_idx is not None, "output_start_idx is None"
        
        input_ids = torch.tensor(input_ids)
        labels = torch.ones_like(input_ids) * -100
        labels[hard_position_type_ids == 3] = input_ids[hard_position_type_ids == 3]
        hpti_out = hard_position_type_ids[output_start_idx:]
        input_ids_out = input_ids[output_start_idx:]
        labels_out = labels[output_start_idx:]
        labels_out[hpti_out <= 1] = input_ids_out[hpti_out <= 1]
        labels[output_start_idx:] = labels_out
        
        # cut to max_len
        max_len = min(self.max_len, len(input_ids))
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len,:max_len]
        labels = labels[:max_len]
        hard_position_type_ids = hard_position_type_ids[:max_len]
        position_ids = position_ids[:max_len]
        # print("done.")
        return input_ids, attention_mask, labels, hard_position_type_ids, position_ids
    
    def collate_fn(self, batch):
        input_ids = [b[0] for b in batch]
        attention_mask = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        hard_position_type_ids = [b[3] for b in batch]
        position_ids = [b[4] for b in batch]
        
        return self.pad_inputs((input_ids, attention_mask, labels, hard_position_type_ids, position_ids), self.tokenizer.pad_token_id)
    
    def pad_inputs(self, batch, pad_token_id=None):
        '''(input_ids:list[tensor], attention_mask:list[tensor, shape=seq*seq], labels:list[tensor])'''
        input_ids, attention_mask, labels, hard_position_type_ids, position_ids = batch[0], batch[1], batch[2], batch[3], batch[4]
        bsz = len(input_ids)
        max_len = max([x.shape[-1] for x in input_ids])
        input_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=pad_token_id) for x in input_ids]).view(bsz, max_len)
        attention_mask = torch.stack([F.pad(x, (max_len - x.shape[-1], 0, 0, max_len - x.shape[-1]), mode='constant', value=0) for x in attention_mask]).view(bsz, 1, max_len, max_len)
        labels = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-100) for x in labels]).view(bsz, max_len) if labels else None
        hard_position_type_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-1) for x in hard_position_type_ids]).view(bsz, max_len)
        position_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=0) for x in position_ids]).view(bsz, max_len)
        
        return input_ids, attention_mask, labels, hard_position_type_ids, position_ids

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.make_input_func(self.data[idx], self.kg, self.tokenizer)

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("/home/cs/yangyuchen/yushengliao/Medical_LLM/FastChat/checkpoints/medical_llama_13b_chatv1.3/checkpoint-4974/")
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.add_tokens(["[DASH]"])
    dst = EntityDataset(data_path='data/kg_HealthCareMagic-20k_train.json', kg_path='data/bios_kg_with_def.csv', tokenizer=tok)
    all_dst = list(dst)