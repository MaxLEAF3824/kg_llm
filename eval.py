import json
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
from rouge import Rouge

rouge = Rouge()

def rougel(row):
    pred = row['output'].strip()
    gt = row['gt'].strip()
    rouge_score = rouge.get_scores(pred, gt)
    row['score'] = rouge_score[0]['rouge-l']['f']
    return row

def eq(row):
    pred = row['output'].strip()
    gt = row['gt'].strip()
    row['score'] = int(pred == gt)
    return row

def main(res_path, metric='eq'):
    res = json.load(open(res_path))
    res_df = pd.DataFrame(res)
    res_df = res_df.apply(eval(metric), axis=1)
    print(res_df['score'].mean())
    
if __name__ == "__main__":
    fire.Fire(main)