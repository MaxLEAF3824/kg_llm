import os
import os.path as osp
import fire
from pprint import pprint
import torch
from lmdeploy import turbomind as tm
from lmdeploy.turbomind.tokenizer import Tokenizer
import json
import jsonlines
import random
from transformers import GenerationConfig,AutoTokenizer,AutoModelForCausalLM
import deepspeed
import pickle
from tqdm import tqdm

class LoadWoInit:
    """Context manager that disable parameter initialization."""

    def __init__(self):
        self.constant_ = torch.nn.init.constant_
        self.zeros_ = torch.nn.init.zeros_
        self.ones_ = torch.nn.init.ones_
        self.uniform_ = torch.nn.init.uniform_
        self.normal_ = torch.nn.init.normal_
        self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        self.kaiming_normal_ = torch.nn.init.kaiming_normal_

    def __enter__(self, *args, **kwargs):
        torch.nn.init.constant_ = lambda *args, **kwargs: None
        torch.nn.init.zeros_ = lambda *args, **kwargs: None
        torch.nn.init.ones_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None

    def __exit__(self, *args, **kwargs):
        torch.nn.init.constant_ = self.constant_
        torch.nn.init.zeros_ = self.zeros_
        torch.nn.init.ones_ = self.ones_
        torch.nn.init.uniform_ = self.uniform_
        torch.nn.init.normal_ = self.normal_
        torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
        torch.nn.init.kaiming_normal_ = self.kaiming_normal_

os.environ['TM_LOG_LEVEL'] = 'ERROR'

PROMPT_TEMPLATE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n##USER:\n{input}\n\n##ASSISTANT:\n{output}"

def get_local_rank():
    return int(os.getenv('LOCAL_RANK', '0'))

def get_rank():
    return int(os.getenv('RANK', '0'))

def get_world_size():
    return int(os.getenv('WORLD_SIZE', '1'))

def prepare_usmle_test_dst():
    usmle_data_path = '/home/cs/yangyuchen/yushengliao/Medical_LLM/data/USMLEdataset/data_clean/questions/US/test.json'
    data = json.load(open(usmle_data_path))
    dst = []
    for d in data:
        input = "<s> " + PROMPT_TEMPLATE.format(input=d['question']+ " Options: " + ', '.join([i+': '+o for (i,o) in d['options'].items()]),output='')
        gt = d['answer']
        dst.append(dict(input=input, gt=gt))
    return dst
    
def prepare_medmcqa_test_dst():
    medmcqa_data_path = '/home/cs/yangyuchen/guoyiqiu/kg_llm/data/medmcqa/dev.json'
    data = list(jsonlines.open(medmcqa_data_path))
    dst = []
    for d in data:
        question = d['question'].strip()
        options = ' Options:' + ', '.join([f"{v}: {d[k]}" for k,v in [('opa','A'),('opb','B'),('opc','C'),('opd','D')]])
        gt = ['A','B','C','D'][int(d['cop']-1)]
        input = "<s> " + PROMPT_TEMPLATE.format(input=question+options,output='')
        dst.append(dict(input=input,gt=gt))
    return dst

def prepare_usmle_test_cot_dst():
    usmle_cot_data_path = '/home/cs/yangyuchen/guoyiqiu/kg_llm/data/cot_kg_usmle_test.json'
    data = json.load(open(usmle_cot_data_path))
    return [dict(input="<s> " + PROMPT_TEMPLATE.format(input=d['input'], output='The related knowledge of the medical entities include:'), gt=d['output']) for d in data]
    
def prepare_medmcqa_test_cot_dst():
    medmcqa_cot_data_path = '/home/cs/yangyuchen/guoyiqiu/kg_llm/data/cot_kg_medmcqa_dev.json'
    data = json.load(open(medmcqa_cot_data_path))
    return [dict(input="<s> " + PROMPT_TEMPLATE.format(input=d['input'], output='The related knowledge of the medical entities include:'), gt=d['output']) for d in data]

def make_test(model, tok, dst, mnt, save_path):
    local_rank = get_local_rank()
    world_size = get_world_size()
    local_start = len(dst)//world_size*local_rank
    local_end = len(dst)//world_size*(local_rank+1) if local_rank != world_size-1 else len(dst)
    local_dst = dst[local_start:local_end]
    
    output_texts = []
    
    for d in tqdm(local_dst, total=len(local_dst)):
        input_ids = tok(d['input'], add_special_tokens=False, return_tensors='pt')['input_ids']
        input_len = input_ids.shape[-1]
        input_ids = input_ids.to(model.module.device)
        output = model.generate(input_ids=input_ids, max_new_tokens=mnt, do_sample=False)
        output_text = tok.decode(output[0,input_len:], skip_special_tokens=True)
        print('output_text: ', output_text)
        output_texts.append(dict(input=d['input'], output=output_text, gt=d['gt']))

    pickle.dump(output_texts, open(f'{save_path}_{local_rank}.pkl', 'wb'))
    torch.distributed.barrier()
    
    if local_rank == 0:
        print("rank 0 collecting results...")
        output_texts = []
        for i in range(world_size):
            output_texts += pickle.load(open(f'{save_path}_{i}.pkl', 'rb'))
            os.remove(f'{save_path}_{i}.pkl')
        json.dump(output_texts, open(save_path, 'w'))
    
def main(model_path='/home/cs/yangyuchen/guoyiqiu/kg_llm/scripts/output/llama_hc_baseline_ft', mnt=64):
    torch.distributed.init_process_group(backend="nccl")
    world_size = get_world_size()
    local_rank = get_local_rank()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    with LoadWoInit():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,).cuda(local_rank)
    
    print("infering usmle...")
    usmle_dst = prepare_usmle_test_dst()
    make_test(model, tokenizer, usmle_dst, mnt, f'{model_path}/usmle_test_result.json')
    print("infering medmcqa...")
    medmcqa_dst = prepare_medmcqa_test_dst()
    make_test(model, tokenizer, medmcqa_dst, mnt, f'{model_path}/medmcqa_test_result.json')
    
    # print("infering cot usmle...")
    # usmle_cot_dst = prepare_usmle_test_cot_dst()
    # make_test(generator, tokenizer, usmle_cot_dst, mnt, f'{model_path}/cot_usmle_test_result.json')
    # print("infering cot medmcqa...")
    # medmcqa_cot_dst = prepare_medmcqa_test_cot_dst()
    # make_test(generator, tokenizer, medmcqa_cot_dst, mnt, f'{model_path}/cot_medmcqa_test_result.json')

if __name__ == '__main__':
    fire.Fire(main)
