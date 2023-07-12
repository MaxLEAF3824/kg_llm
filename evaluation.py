import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def evaluation_basic(model, tok, train_data_path, kg_path = "data/umls_kg_filter.csv"):
    train_dst = json.load(open(train_data_path, "r"))
    et = {e:t for es,ts in zip([d['input_entities']+d['output_entities'] for d in train_dst], [d['input_triplets']+d['output_triplets'] for d in train_dst]) for e,t in zip(es,ts)}
    kg = pd.read_csv(kg_path).to_dict(orient='records')
    dash_token = "[DASH]"
    
    all_acc = []
    for e in et.keys():
        t = et[e]
        bsz = 16
        e_acc = 0
        e_res = []
        for tid in t:
            tri = kg[tid]
            prompts = [f"{tri['source']}  {dash_token} {tri['edge']}", f"{tri['target']}  {dash_token} {tri['edge']}"]
            tok.padding_side = 'left'
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
            max_new_tokens = 30
            
            for prompt in prompts:
                inp = tok(prompt, return_tensors="pt")
                inp_len = inp['input_ids'].shape[1]
                print(inp['input_ids'].shape)
                res = model.generate(inp['input_ids'].to(model.device), attention_mask=inp['attention_mask'].to(model.device), max_new_tokens=max_new_tokens)
                res = tok.batch_decode(res[:,inp_len:], skip_special_tokens=True)
            e_res.extend(res)
        
        for i, tids in enumerate(t):
            tri = kg[tids]
            print(f"tri target:{tri['target']} e_res:{e_res[i*2]} tri source:{tri['source']} e_res:{e_res[i*2+1]}")
            if tri['target'] in e_res[i*2] or tri['source'] in e_res[i*2+1]:
                e_acc += 1
        e_acc /= len(t)
        all_acc.append(e_acc)
        print(f"{e} acc:{e_acc}")
    print(f"avg acc:{sum(all_acc)/len(all_acc)}")
            
            
if __name__ == "__main__":
    mt_path = "/home/cs/yangyuchen/guoyiqiu/kg_llm/output/full_book_13b_bsz1_epoch2_lr1e-05"
    mt_path = "/home/cs/yangyuchen/guoyiqiu/my_models/gpt2"
    mt_path = "/home/cs/yangyuchen/yushengliao/Medical_LLM/vicuna-7b"
    model = AutoModelForCausalLM.from_pretrained(mt_path).cuda()
    tok = AutoTokenizer.from_pretrained(mt_path)
    # mt_path = "/home/cs/yangyuchen/guoyiqiu/my_models/gpt2"
    train_data_path = "data/kg_instruction_10.json"
    evaluation_basic(model, tok, train_data_path)