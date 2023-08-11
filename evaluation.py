import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
import pytorch_lightning as pl
import fastlcs
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def evaluation_stage1():
    mt_path = "/home/cs/yangyuchen/yushengliao/Medical_LLM/vicuna-7b"
    # model = AutoModelForCausalLM.from_pretrained(mt_path).cuda()
    # tok = AutoTokenizer.from_pretrained(mt_path)
    data_path = "data/kg_instruction_10.json"
    kg_path = "data/umls_kg_filter_count_5_with_def_len_100.csv"
    kg = pd.read_csv(kg_path)
    data = json.load(open(data_path))
    all_triplets = [tid for i in data for triplets in i['input_triplets']+i['output_triplets'] for tid in triplets]
    all_triplets = list(set(all_triplets))
    print(len(all_triplets))
    
            
            
if __name__ == "__main__":
    mt_path = "/home/cs/yangyuchen/yushengliao/Medical_LLM/vicuna-7b"
    # model = AutoModelForCausalLM.from_pretrained(mt_path).cuda()
    # tok = AutoTokenizer.from_pretrained(mt_path)
    data_path = "data/kg_instruction_10.json"
    kg_path = "data/umls_kg_filter_count_5_with_def_len_100.csv"
    evaluation_stage1()