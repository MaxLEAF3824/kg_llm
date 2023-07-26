from click import group
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModelForCausalLM
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
from accelerate import Accelerator
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import uuid



class EntityDataset(Dataset):
    def __init__(self, data_path: str, kg_path : str, tokenizer: Tokenizer, size=None, max_len=2048, from_pickle=None,
                 add_dash=True, dash_token='[DASH]', dump=False, dump_name=None, dump_dir="data", lazy=False, *args, **kwargs):
        self.prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n##USER:\n{input}\n\n##ASSISTANT:\n{output}"
        self.max_len = max_len
        self.add_dash = add_dash
        self.dash_token = dash_token
        self.tokenizer = tokenizer
        self.kg = pd.read_csv(kg_path).to_dict(orient='records')
        self.data = json.load(open(data_path))
        self.data = self.data[:size] if size else self.data
        if not lazy:
            if from_pickle:
                self.input_ids, self.attention_mask, self.labels, self.hard_position_type_ids, self.prompts = pickle.load(open(from_pickle, "rb"))
                print(f"Loaded dataset from pickle file {from_pickle}")
            else:
                self.input_ids, self.attention_mask, self.labels, self.hard_position_type_ids, self.prompts = self.make_inputs()
                if dump:
                    if dump_name is None:
                        dump_name = f"EntityDataset_{len(self)}_{str(uuid.uuid4().int)[:8]}"
                    dump_path = f"{dump_dir}/{dump_name}.pkl"
                    pickle.dump((self.input_ids, self.attention_mask, self.labels, self.hard_position_type_ids, self.prompts), open(dump_path, "wb"))
                    print(f"dump dataset to pickle file {dump_path}")
        else:
            print("lazy loading...")
            self.__getitem__ = self.lazy_getitem
        print(f"Loaded dataset with {len(self)} elements")
    
    def make_inputs(self):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        hard_position_type_ids_list = []
        prompt_list = []

        # 多进程
        shared_data = self.data
        shared_kg = self.kg
        # shared_data = manager.list(self.data)
        # shared_kg = manager.list(self.kg)
        make_input_func = partial(self.make_input_func, kg=shared_kg, tok=self.tokenizer)
        for output in tqdm(pool.imap_unordered(make_input_func, shared_data),total=len(shared_data)):
            input_ids_list.append(output[0])
            attention_mask_list.append(output[1])
            labels_list.append(output[2])
            hard_position_type_ids_list.append(output[3])
            prompt_list.append(output[4])

        # 多线程        
        # import concurrent.futures
        # process_num = 2
        # make_input_func = partial(self.make_input_func, kg=self.kg, tok=self.tokenizer)
        # with concurrent.futures.ThreadPoolExecutor(process_num) as executor:
        #     future_to_input = {executor.submit(make_input_func, ins): ins for ins in self.data}
        #     for future in tqdm(concurrent.futures.as_completed(future_to_input),total=len(self.data)):
        #         ins = future_to_input[future]
        #         try:
        #             output = future.result()
        #             input_ids_list.append(output[0])
        #             attention_mask_list.append(output[1])
        #             labels_list.append(output[2])
        #             hard_position_type_ids_list.append(output[3])
        #             prompt_list.append(output[4])
        #         except Exception as exc:
        #             print(f'An exception occurred: {exc}')
        
        # 单进程
        # make_input_func = partial(self.make_input_func, kg=self.kg, tok=self.tokenizer)
        # for ins in tqdm(self.data, total=len(self.data)):
        #     output = make_input_func(ins)
        #     input_ids_list.append(output[0])
        #     attention_mask_list.append(output[1])
        #     labels_list.append(output[2])
        #     hard_position_type_ids_list.append(output[3])
        #     prompt_list.append(output[4])
        
        return input_ids_list, attention_mask_list, labels_list, hard_position_type_ids_list, prompt_list
    
    def make_input_func(self, ins, kg, tok):
        # tok = deepcopy(tok)
        all_text = self.prompt_template.format(input=ins['input'], output=ins['output'])
        all_text = all_text.strip() + f" {tok.eos_token}"
        # print(f"all_text:{all_text}")
        inp = tok(all_text)
        input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
        et = list({e:t for e,t in zip(ins['input_entities']+ins['output_entities'], ins['input_triplets']+ins['output_triplets']) if len(t) > 0}.items())
        et = sorted(et, key=lambda et: - len(et[0])) # 按entity的长度从长到短排序
        # print(f"et:{et}")
        # print(f"input_ids:{input_ids}")

        # 识别input_ids中的entity并替换为0,-1,-2...
        for i,(e,t) in enumerate(et):
            e_ids = tok(e, add_special_tokens=False)['input_ids']
            # print(f"{e} e_ids:{e_ids}")
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
        
        # print(f"input_ids with entities labeled:{input_ids}")

        # 拓展input_ids并计算每个token的hard_position_type_id和group_id, 0代表non-entity tokens，1代表entity tokens, 2代表triplet tokens, 3代表triplet target tokens
        hard_position_type_ids = []
        group_ids = []
        new_input_ids = []
        idx = 0
        group_idx = 0
        while idx < len(input_ids):
            if input_ids[idx] <= 0:
                e, t = et[-input_ids[idx]]
                triplets = [kg[tid] for tid in t]
                e_ids = tok(e, add_special_tokens=False)['input_ids']
                new_input_ids.extend(e_ids)
                hard_position_type_ids.extend([1] * len(e_ids))
                group_ids.extend([-1] * len(e_ids))
                triplets = [random.choice(triplets)]
                for triplet in triplets:
                    tri_prompt = f" [DASH] " if self.add_dash else f" "
                    tri_target = f"{triplet['source']} {triplet['edge']} {triplet['target']}"
                    tri_prompt_id = tok(tri_prompt, add_special_tokens=False)['input_ids']
                    tri_target_id = tok(tri_target, add_special_tokens=False)['input_ids']
                    new_input_ids.extend(tri_prompt_id)
                    new_input_ids.extend(tri_target_id)
                    hard_position_type_ids.extend([2] * len(tri_prompt_id))
                    hard_position_type_ids.extend([3] * len(tri_target_id))
                    group_ids.extend([group_idx] * len(tri_prompt_id))
                    group_ids.extend([group_idx] * len(tri_target_id))
                    group_idx += 1
                idx += 1
            else:
                new_input_ids.append(input_ids[idx])
                hard_position_type_ids.append(0) # 0 means original text
                group_ids.append(-1) # -1 means original text
                idx += 1
        
        
        seq_len = len(new_input_ids)
        assert seq_len == len(hard_position_type_ids)
        input_ids = new_input_ids
        # print(f"seq_len:{seq_len}")
        # print(f"new_input_ids:{new_input_ids}")
        # print(f"new_input:{tok.decode(new_input_ids)}")
        # print(f"hard_position_type_ids:{hard_position_type_ids} length:{len(hard_position_type_ids)}")

        # 利用hard_position_type_ids计算attention_mask_map, shape为seq_len*seq_len
        # 0代表non-entity tokens，1代表entity tokens, 2代表triplet tokens, 3代表triplet target tokens
        attention_mask = torch.zeros(seq_len,seq_len, dtype=torch.bool)

        for i in (range(seq_len)):
            for j in range(i): # i>j
                if hard_position_type_ids[i] == 0:
                    if hard_position_type_ids[j] == 0 or hard_position_type_ids[j] == 1:
                        attention_mask[i,j] = 1
                elif hard_position_type_ids[i] == 1:
                    if hard_position_type_ids[j] == 0 or hard_position_type_ids[j] == 1:
                        attention_mask[i,j] = 1
                elif hard_position_type_ids[i] == 2:
                    if hard_position_type_ids[j] == 0:
                        attention_mask[i,j] = 1
                    if hard_position_type_ids[j] == 2 and group_ids[i] == group_ids[j]:
                        attention_mask[i,j] = 1
                elif hard_position_type_ids[i] == 3:
                    if hard_position_type_ids[j] == 0:
                        attention_mask[i,j] = 1
                    if hard_position_type_ids[j] == 2 and group_ids[i] == group_ids[j]:
                        attention_mask[i,j] = 1
                    if hard_position_type_ids[j] == 3 and group_ids[i] == group_ids[j]:
                        attention_mask[i,j] = 1
        
        # print(f"attention_mask:{attention_mask.shape}")
        
        blank_prompt = self.prompt_template.format(input="",output="")
        begin_out_ids = tok(blank_prompt)['input_ids'][-5:]
        begin_out_idxs = [i for i in range(len(input_ids)) if input_ids[i:i+len(begin_out_ids)] == begin_out_ids]
        if begin_out_idxs == []:
            print(f"output_start_ids:{begin_out_ids}\n{tok.batch_decode(begin_out_ids)}")
            print(f"input_ids:{input_ids}\n{tok.batch_decode(input_ids)}")
            raise ValueError("begin_out_idxs is empty")
        output_start_idx = begin_out_idxs[0] + len(begin_out_ids)
        input_ids = torch.tensor(input_ids)
        labels = torch.ones_like(input_ids) * -100
        
        for i in range(len(labels)):
            if hard_position_type_ids[i] == 3:
                labels[i] = input_ids[i]
            elif i > output_start_idx and (hard_position_type_ids[i] == 0 or hard_position_type_ids[i] == 1):
                labels[i] = input_ids[i]

        hard_position_type_ids = torch.tensor(hard_position_type_ids)
        
        # cut to max_len
        max_len = min(self.max_len, len(input_ids))
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len,:max_len]
        labels = labels[:max_len]
        hard_position_type_ids = hard_position_type_ids[:max_len]
        
        prompt = tok.decode(input_ids)
        # print("done.")
        return input_ids, attention_mask, labels, hard_position_type_ids, prompt
    
    def collate_fn(self, batch):
        input_ids = [b[0] for b in batch]
        attention_mask = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        hard_position_type_ids = [b[3] for b in batch]
        
        return self.pad_inputs((input_ids, attention_mask, labels, hard_position_type_ids), self.tokenizer.pad_token_id)
    
    def pad_inputs(self, batch, pad_token_id=None):
        '''(input_ids:list[tensor], attention_mask:list[tensor, shape=seq*seq], labels:list[tensor])'''
        input_ids, attention_mask, labels, hard_position_type_ids = batch[0], batch[1], batch[2], batch[3]
        bsz = len(input_ids)
        max_len = max([x.shape[-1] for x in input_ids])
        input_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=pad_token_id) for x in input_ids]).view(bsz, max_len)
        attention_mask = torch.stack([F.pad(x, (max_len - x.shape[-1], 0, 0, max_len - x.shape[-1]), mode='constant', value=0) for x in attention_mask]).view(bsz, 1, max_len, max_len)
        labels = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-100) for x in labels]).view(bsz, max_len) if labels else None
        hard_position_type_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-1) for x in hard_position_type_ids]).view(bsz, max_len)
        
        return input_ids, attention_mask, labels, hard_position_type_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx], self.hard_position_type_ids[idx]
    
    def lazy_getitem(self, idx):
        ins = self.data[idx]
        input_ids, attention_mask, labels, hard_position_type_ids, prompt = self.make_input_func(ins, self.tokenizer)
        return input_ids, attention_mask, labels, hard_position_type_ids

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/cs/yangyuchen/yushengliao/Medical_LLM/FastChat/checkpoints/medical_llama_13b_chatv1.3/checkpoint-4974/")
    # 调整tokenizer
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    dash_token = "[DASH]"
    tok.add_tokens([dash_token])
    # manager = Manager()
    process_num = 24
    pool = Pool(process_num)
    print(f"Pool created with {process_num} process")
    dst = EntityDataset(data_path='data/kg_chat_usmle_10178.json', kg_path='data/umls_kg_filter_count_5_with_def_len_100.csv', 
                        add_dash=True, tokenizer=tok, max_len=8192, dump=True, dump_name="8192_ep_0_new_dash_front", dump_dir="data/EntityDataset_chat_usmle")
    dl = DataLoader(dst, batch_size=4, shuffle=True, collate_fn=dst.collate_fn)
    for d in dl:
        print(d)
        break
    pool.close()
    