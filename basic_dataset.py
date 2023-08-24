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



class BasicDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, size=None, max_len=2048, from_pickle=None, dash_token='[DASH]', dump=False, dump_name=None, dump_dir="data", lazy=False, *args, **kwargs):
        self.prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n##USER:\n{input}\n\n##ASSISTANT:\n{output}"
        self.max_len = max_len
        self.dash_token = dash_token
        self.tokenizer = tokenizer
        self.data = json.load(open(data_path))
        self.data = self.data[:size] if size else self.data
        self.length = len(self.data)
        print(f"Loaded dataset with {len(self)} elements")
        
    
    def make_inputs(self):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        prompt_list = []
        # 单进程
        make_input_func = partial(self.make_input_func, tok=self.tokenizer)
        for ins in tqdm(self.data, total=len(self.data)):
            output = make_input_func(ins)
            input_ids_list.append(output[0])
            attention_mask_list.append(output[1])
            labels_list.append(output[2])
            prompt_list.append(output[3])
        
        return input_ids_list, attention_mask_list, labels_list, prompt_list
    
    def make_input_func(self, ins, tok):
        # tok = deepcopy(tok)
        all_text = self.prompt_template.format(input=ins['input'], output=ins['output'])
        all_text = all_text.strip() + f" {tok.eos_token}"
        # print(f"all_text:{all_text}")
        inp = tok(all_text)
        input_ids, attention_mask = inp['input_ids'], inp['attention_mask']

        blank_prompt = self.prompt_template.format(input="",output="")
        begin_out_ids = tok(blank_prompt)['input_ids'][-5:]
        begin_out_idxs = [i for i in range(len(input_ids)) if input_ids[i:i+len(begin_out_ids)] == begin_out_ids]
        if begin_out_idxs == []:
            print(f"output_start_ids:{begin_out_ids}\n{tok.batch_decode(begin_out_ids)}")
            print(f"input_ids:{input_ids}\n{tok.batch_decode(input_ids)}")
            raise ValueError("begin_out_idxs is empty")
        output_start_idx = begin_out_idxs[0] + len(begin_out_ids)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = input_ids.clone()
        labels[:output_start_idx] = -100
        
        
        max_len = min(self.max_len, len(input_ids))
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]
        
        return input_ids, attention_mask, labels
    
    def collate_fn(self, batch):
        input_ids = [b[0] for b in batch]
        attention_mask = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        
        return self.pad_inputs((input_ids, attention_mask, labels), self.tokenizer.pad_token_id)
    
    def pad_inputs(self, batch, pad_token_id=None):
        '''(input_ids:list[tensor], attention_mask:list[tensor, shape=seq*seq], labels:list[tensor])'''
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        bsz = len(input_ids)
        max_len = max([x.shape[-1] for x in input_ids])
        input_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=pad_token_id) for x in input_ids]).view(bsz, max_len)
        attention_mask = torch.stack([F.pad(x, (max_len - x.shape[-1], 0,), mode='constant', value=0) for x in attention_mask]).view(bsz, max_len)
        labels = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-100) for x in labels]).view(bsz, max_len) if labels else None
        
        return input_ids, attention_mask, labels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ins = self.data[idx]
        input_ids, attention_mask, labels = self.make_input_func(ins, self.tokenizer)
        return input_ids, attention_mask, labels


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/cs/yangyuchen/yushengliao/Medical_LLM/FastChat/checkpoints/medical_llama_13b_chatv1.3/checkpoint-4974/")
    # 调整tokenizer
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    dst = BasicDataset(data_path='data/usmle_train.json', tokenizer=tok)
    print(dst[0])