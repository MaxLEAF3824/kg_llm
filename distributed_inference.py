import sys
sys.path.append("/home/cs/yangyuchen/guoyiqiu/gpt_re")
from model import LLM
import time
import pytorch_lightning as pl
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
import json
import os
import jsonlines
import typing
from pathlib import Path
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
torch.set_float32_matmul_precision('medium')


def pad_inputs(batch, pad_token_id=None):
    '''(input_ids:list[tensor], attention_mask:list[tensor], labels:list[tensor])'''
    input_ids, attention_mask = batch[0], batch[1]
    labels = batch[2] if len(batch) == 3 else None
    max_len = max([x.shape[-1] for x in input_ids])
    # align right for gpt model
    input_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=pad_token_id) for x in input_ids]).squeeze().unsqueeze(0)
    attention_mask = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=0) for x in attention_mask]).squeeze().unsqueeze(0)
    labels = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-100) for x in labels]).squeeze() if labels else None
    return input_ids, attention_mask, labels

PROMPT_TEMPLATE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n##USER:\n{input}\n\n##ASSISTANT:\n{output}"

class USMLETest(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, size=None, max_len=1024, *args, **kwargs,):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data = json.load(open(data_path))
        self.data = self.data[:size] if size else self.data
        self.input_ids, self.attention_mask, self.labels, self.prompts = self.make_inputs(self.data)
        print(f"Loaded dataset with {len(self)} elements")
    def make_inputs(self, data):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        prompt_list = []
        for d in data:
            question = d['question'].strip()
            options = '\nOptions:' + ' '.join([f"{k}: {d['options'][k]}" for k in d['options']])
            answer = d['answer']
            prompt = PROMPT_TEMPLATE.format(input=question+options, output=answer)
            res = self.tokenizer(prompt, return_tensors='pt')
            input_ids, attention_mask = res.input_ids, res.attention_mask
            if input_ids.shape[-1] > self.max_len:
                continue
            labels = -100 * torch.ones_like(input_ids)
            labels[:,-1] = input_ids[:,-1]
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            prompt_list.append(prompt)
        return input_ids_list, attention_mask_list, labels_list, prompt_list
    def collate_fn(self, batch):
        input_ids = [b[0] for b in batch]
        attention_mask = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        return pad_inputs((input_ids, attention_mask, labels), self.tokenizer.pad_token_id)
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

def my_predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    with torch.no_grad():
        input_ids, attention_mask = batch[0], batch[1]
        label = batch[2]
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128, do_sample=False)
    output = output[0, input_ids.shape[-1]-1:]
    input_text = self.tokenizer.decode(input_ids.squeeze())
    output_text = self.tokenizer.decode(output.squeeze())
    label_text = self.tokenizer.decode(label[0,-1])
    return dict(question=input_text,
                text=output_text,
                answer=label_text)

class MedMCQATest(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, size=None, max_len=1024, *args, **kwargs,):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data = json.load(open(data_path))
        self.data = self.data[:size] if size else self.data
        self.input_ids, self.attention_mask, self.labels, self.prompts = self.make_inputs(self.data)
        print(f"Loaded dataset with {len(self)} elements")
    def make_inputs(self, data):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        prompt_list = []
        for d in data:
            question = d['question'].strip()
            options = '\nOptions:' + ' '.join([f"{k}: {d['options'][k]}" for k in d['options']])
            answer = d['answer']
            prompt = PROMPT_TEMPLATE.format(input=question+options, output=answer)
            res = self.tokenizer(prompt, return_tensors='pt')
            input_ids, attention_mask = res.input_ids, res.attention_mask
            if input_ids.shape[-1] > self.max_len:
                continue
            labels = -100 * torch.ones_like(input_ids)
            labels[:,-1] = input_ids[:,-1]
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            prompt_list.append(prompt)
        return input_ids_list, attention_mask_list, labels_list, prompt_list
    def collate_fn(self, batch):
        input_ids = [b[0] for b in batch]
        attention_mask = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        return pad_inputs((input_ids, attention_mask, labels), self.tokenizer.pad_token_id)
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

def my_predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    mnt_step = 256
    input_ids, attention_mask,label = batch[0][:,:-1], batch[1][:,:-1], batch[2][0,-1]
    input_text = self.tokenizer.decode(input_ids.squeeze())
    label_text = self.tokenizer.decode(label)
    with torch.no_grad():
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=mnt_step, do_sample=False)
    output = output[0, input_ids.shape[-1]:]
    output_text = self.tokenizer.decode(output.squeeze())
    return dict(question=input_text,
                text=output_text,
                answer=label_text)

if __name__=="__main__":
    CUDA_VISIBLE_DEVICES = [0,3,4,5,6,7]
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in CUDA_VISIBLE_DEVICES])
    mt_path = '/home/cs/yangyuchen/guoyiqiu/kg_llm/output/chat_usmle_kg_sample_one_new'
    model = AutoModelForCausalLM.from_pretrained(mt_path).half()
    tok = AutoTokenizer.from_pretrained(mt_path)
    mt = LLM.from_mt(model, tok)
    data_path = '/home/cs/yangyuchen/yushengliao/Medical_LLM/data/USMLEdataset/data_clean/questions/US/test.json'
    dst = USMLETest(data_path, tok, max_len=2048)
    mt.set_func("predict_step", my_predict_step)
    trainer = pl.Trainer(devices=list(range(len(CUDA_VISIBLE_DEVICES))), strategy="ddp", precision=16, logger=False)
    # trainer = pl.Trainer(devices=[0,1], precision=16, logger=False)
    # res = trainer.test(mt, dst)
    res = trainer.predict(mt, dst)
    import json
    js = json.load(open(data_path))
    json.dump(res, open(f"{mt_path}/answer.json", 'w'))