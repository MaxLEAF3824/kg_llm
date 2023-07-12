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
from multiprocessing import Pool, cpu_count
from accelerate import Accelerator
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import uuid



class EntityDataset(Dataset):
    def __init__(self, data_path: str, kg_path : str, tokenizer: Tokenizer, size=None, max_len=1024, from_pickle=None, dash_token='[DASH]', dump=False, dump_dir="data", *args, **kwargs):
        self.prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n##USER:\n{input}\n\n##ASSISTANT:\n{output}"
        self.max_len = max_len
        self.dash_token = dash_token
        self.tokenizer = tokenizer
        self.kg = pd.read_csv(kg_path).to_dict(orient='records')
        self.data = json.load(open(data_path))
        self.data = self.data[:size] if size else self.data
        if from_pickle:
            self.input_ids, self.attention_mask, self.labels, self.hard_position_type_ids, self.prompts = pickle.load(open(from_pickle, "rb"))
            print(f"Loaded dataset from pickle file {from_pickle}")
        else:
            self.input_ids, self.attention_mask, self.labels, self.hard_position_type_ids, self.prompts = self.make_inputs()
            if dump:
                dump_path = f"{dump_dir}/EntityDataset_{len(self)}_{str(uuid.uuid4().int)[:8]}.pkl"
                pickle.dump((self.input_ids, self.attention_mask, self.labels, self.hard_position_type_ids, self.prompts), open(dump_path, "wb"))
                print(f"dump dataset to pickle file {dump_path}")
        
        print(f"Loaded dataset with {len(self)} elements")
    
    def make_inputs(self):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        hard_position_type_ids_list = []
        prompt_list = []

        # 多进程
        # process_num = 64
        # pool = Pool(process_num)
        # print(f"Pool created with {process_num} process")
        # shared_data = self.data
        # shared_kg = self.kg
        # from multiprocessing import Manager
        # manager = Manager()
        # shared_data = manager.list(self.data)
        # shared_kg = manager.list(self.kg)
        # make_input_func = partial(self.make_input_func, kg=shared_kg, tok=self.tokenizer)
        # for output in tqdm(pool.imap(make_input_func, shared_data),total=len(shared_data)):
        #     input_ids_list.append(output[0])
        #     attention_mask_list.append(output[1])
        #     labels_list.append(output[2])
        #     hard_position_type_ids_list.append(output[3])
        #     prompt_list.append(output[4])
        # pool.close()

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
        make_input_func = partial(self.make_input_func, kg=self.kg, tok=self.tokenizer)
        for ins in tqdm(self.data, total=len(self.data)):
            output = make_input_func(ins)
            input_ids_list.append(output[0])
            attention_mask_list.append(output[1])
            labels_list.append(output[2])
            hard_position_type_ids_list.append(output[3])
            prompt_list.append(output[4])
        
        return input_ids_list, attention_mask_list, labels_list, hard_position_type_ids_list, prompt_list
    
    def make_input_func(self, ins, kg, tok):
        # tok = deepcopy(tok)
        all_text = self.prompt_template.format(input=ins['input'], output=ins['output'])
        # print(f"all_text:{all_text}")
        inp = tok(all_text)
        input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
        et = list({e:random.choice(t) for e,t in zip(ins['input_entities']+ins['output_entities'], ins['input_triplets']+ins['output_triplets']) if len(t) > 0}.items())
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

        # 拓展input_ids并计算每个token的hard_position_type_id, 0代表non-entity tokens，1代表entity tokens, 2代表triplet tokens, 3代表triplet target tokens
        hard_position_type_ids = []
        new_input_ids = []
        idx = 0
        while idx < len(input_ids):
            if input_ids[idx] <= 0:
                e, t = et[-input_ids[idx]]
                # tid = random.choice(t)
                # et[i][1].remove(tid)
                tid = t
                triplet = kg[tid]
                # print(f"e:{e}\ntriplet source: {triplet['source']}\ntriplet edge: {triplet['edge']}\ntriplet target: {triplet['target']}")
                if e.lower() in triplet['source'].lower():
                    e_idx = triplet['source'].lower().index(e.lower())
                    source = triplet['source'][:e_idx] + e + triplet['source'][e_idx+len(e):]
                    tri_prompt = f"{source}  {self.dash_token} {triplet['edge']}"
                    tri_target = triplet['target']
                elif e.lower() in triplet['target'].lower():
                    e_idx = triplet['target'].lower().index(e.lower())
                    t = triplet['target'][:e_idx] + e + triplet['target'][e_idx+len(e):]
                    tri_prompt = f"{t}  {self.dash_token} {triplet['edge']}"
                    tri_target = triplet['source']
                else:
                    raise ValueError("entity not in source and target")
                assert e in tri_prompt
                entity_ids = tok(e, add_special_tokens=False)['input_ids']
                prompt_ids = tok(tri_prompt[:tri_prompt.index(e)].strip(), add_special_tokens=False)['input_ids'] + entity_ids + tok(tri_prompt[tri_prompt.index(e)+len(e):].strip(), add_special_tokens=False)['input_ids']
                target_ids = tok(tri_target, add_special_tokens=False)['input_ids']
                # print(f"entity:{entity_ids}\n{tok.batch_decode(entity_ids)}\nprompt:{prompt_ids}\n{tok.batch_decode(prompt_ids)}\ntarget:{target_ids}\n{tok.batch_decode(target_ids)}")
                new_input_ids.extend(prompt_ids + target_ids)
                triplet_hard_type_ids = [2 for i in range(len(prompt_ids))] + [3 for i in range(len(target_ids))] # 2 means triplet text, 3 means target text
                entity_relative_start_idxs = [i for i in range(len(prompt_ids)-len(entity_ids)) if prompt_ids[i:i+len(entity_ids)] == entity_ids]
                assert len(entity_relative_start_idxs) != 0
                entity_relative_start_idx = entity_relative_start_idxs[0]
                for i in range(entity_relative_start_idx, entity_relative_start_idx+len(entity_ids)):
                    triplet_hard_type_ids[i] = 1 # 1 means entity text
                hard_position_type_ids.extend(triplet_hard_type_ids)
                idx += 1
            else:
                new_input_ids.append(input_ids[idx])
                hard_position_type_ids.append(0) # 0 means original text
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
        # type0可以看见type0和type1, type1可以看见type0,type1, type2可以看见type1,type2和type3, type3可以看见type1,type2和type3
        attention_mask = torch.tril(torch.ones(seq_len,seq_len))
        for i in (range(seq_len)):
            for j in range(i): # i>j
                if hard_position_type_ids[i] == 0:
                    if hard_position_type_ids[j] == 2:
                        attention_mask[i,j] = 0
                elif hard_position_type_ids[i] == 1:
                    if hard_position_type_ids[j] == 2:
                        attention_mask[i,j] = 0
                else:
                    if hard_position_type_ids[j] == 0:
                        attention_mask[i,j] = 0
        
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
        labels = input_ids.clone()
        labels[:output_start_idx] = -100
        
        for i in range(len(hard_position_type_ids)):
            if hard_position_type_ids[i] == 3:
                labels[i] = input_ids[i]
        hard_position_type_ids = torch.tensor(hard_position_type_ids)
        
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
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx], self.hard_position_type_ids[idx]

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/cs/yangyuchen/yushengliao/Medical_LLM/FastChat/checkpoints/medical_llama_13b_chatv1.3/checkpoint-4974/")
    # 调整tokenizer
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    dash_token = "[DASH]"
    tok.add_tokens([dash_token])
    accelerator = Accelerator()
    dst = EntityDataset(data_path='data/kg_instruction_1000.json', kg_path='data/umls_kg_filter_count_5.csv', tokenizer=tok, max_len=1024, from_pickle="/home/cs/yangyuchen/guoyiqiu/kg_llm/data/EntityDataset_1k/ep_0.pkl")
    dl = DataLoader(dst, batch_size=1, shuffle=True, collate_fn=dst.collate_fn)
    dl = accelerator.prepare(dl)
    for d in dl:
        print(d)
        break