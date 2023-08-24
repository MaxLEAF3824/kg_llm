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

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,6,7'
PROMPT_TEMPLATE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n##USER:\n{input}\n\n##ASSISTANT:\n{output}"

def prepare_usmle_test_dst(prompt=''):
    usmle_data_path = '/home/cs/yangyuchen/yushengliao/Medical_LLM/data/USMLEdataset/data_clean/questions/US/test.json'
    data = json.load(open(usmle_data_path))
    dst = []
    for d in data:
        input = "<s> " + PROMPT_TEMPLATE.format(input=d['question']+ " Options: " + ', '.join([i+': '+o for (i,o) in d['options'].items()]),output=prompt)
        gt = d['answer']
        dst.append(dict(input=input, gt=gt))
    return dst
    
def prepare_medmcqa_test_dst(prompt=''):
    medmcqa_data_path = '/home/cs/yangyuchen/guoyiqiu/kg_llm/data/medmcqa/dev.json'
    data = list(jsonlines.open(medmcqa_data_path))
    dst = []
    for d in data:
        if d['choice_type'] != 'multi':
            question = d['question'].strip()
            options = ' Options:' + ', '.join([f"{v}: {d[k]}" for k,v in [('opa','A'),('opb','B'),('opc','C'),('opd','D')]])
            gt = ['A','B','C','D'][int(d['cop']-1)]
            input = "<s> " + PROMPT_TEMPLATE.format(input=question+options,output=prompt)
            dst.append(dict(input=input,gt=gt))
    return dst

def make_test(generator, tokenizer, dst, mnt, save_path):
    input_ids_list = [tokenizer.encode(d['input']) for d in dst]
    output = generator.stream_infer(session_id=list(range(len(input_ids_list))),input_ids=input_ids_list, request_output_len=mnt, top_p=1, top_k=1, step=0)
    output = list(output)
    output_ids = [o[0] for o in output[0]]
    output_texts = [tokenizer.decode(o) for o in output_ids]
    json.dump([{"input":i['input'],"output":o,"gt":i['gt']} for i,o in zip(dst,output_texts)], open(save_path, 'w'), indent=4)
    return output_texts

def main(model_path='/home/cs/yangyuchen/guoyiqiu/kg_llm/lmdeploy/llama', prompt='', mnt=128):
    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = Tokenizer(tokenizer_model_path)
    tm_model = tm.TurboMind(model_path, eos_id=tokenizer.eos_token_id)
    generator = tm_model.create_instance()
    
    usmle_dst = prepare_usmle_test_dst(prompt=prompt)
    medmcqa_dst = prepare_medmcqa_test_dst(prompt=prompt)
    print("infering usmle...")
    make_test(generator, tokenizer, usmle_dst, mnt, f'{model_path}/usmle_test_result.json')
    print("infering medmcqa...")
    make_test(generator, tokenizer, medmcqa_dst, mnt, f'{model_path}/medmcqa_test_result.json')

if __name__ == '__main__':
    fire.Fire(main)
