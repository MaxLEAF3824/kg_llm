from transformers import AutoTokenizer, AutoModelForTokenClassification
import jsonlines
import pandas as pd
import random
import torch
from tqdm.auto import tqdm
import ujson as json
import re
import bisect
import pdb
import traceback

slice_start = 44000
size = 1000
instructions = list(jsonlines.open('data/instruction_dataall.jsonl'))[slice_start:slice_start+size]
df = pd.read_csv("data/umls_kg_filter.csv")

mt_path = "/mnt/workspace/guoyiqiu/coding/huggingface/my_models/RohanVB_umlsbert_ner"

ner_model = AutoModelForTokenClassification.from_pretrained(mt_path)
ner_tok = AutoTokenizer.from_pretrained(mt_path)
ner_model = ner_model.cuda()
ner_model.eval()

error_count = 0

def batch_list_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def batch_ner(prompts, ner_model, ner_tok, max_len=512):
    inp = ner_tok(prompts, return_tensors='pt',padding=True).to(ner_model.device)
    
    if inp['input_ids'].shape[1] > max_len:
        inp['input_ids'] = inp['input_ids'][:, :max_len]
        inp['attention_mask'] = inp['attention_mask'][:, :max_len]
        inp['token_type_ids'] = inp['token_type_ids'][:, :max_len]
    
    
    with torch.no_grad():
        batch_ner_output = ner_model(**inp).logits.argmax(-1)
    
    batch_entities = []
    for input_ids, ner_output, prompt in zip(inp['input_ids'], batch_ner_output, prompts):
        word_idxs = [idx for idx, i in enumerate(input_ids.tolist()) if not ner_tok.convert_ids_to_tokens(i).startswith("##")]
        words = [ner_tok.decode(input_ids[idx: word_idxs[i+1]]) for i, idx in enumerate(word_idxs[:-1])] + [ner_tok.decode(input_ids[word_idxs[-1]:])]
        # print(prompt)
        # print(words)
        # print(ner_output)

        # ner_output[ner_output>3] = ner_output[ner_output>3] - 3
        entities = []
        cur_entity = []
        cur_ner_id = 0
        idx = 0
        while idx < len(ner_output):
            ner_id = ner_output[idx].item()
            if ner_id == 0:
                idx += 1
                continue
            if ner_id > 3:
                if not cur_entity:
                    cur_entity = [idx]
                    cur_ner_id = ner_id
                else:
                    cur_entity_start_idx = cur_entity[0]
                    cur_entity_end_idx = cur_entity[-1]
                    cur_entity_start_word_idx = bisect.bisect_right(word_idxs, cur_entity_start_idx) - 1
                    cur_entity_end_word_idx = bisect.bisect_right(word_idxs, cur_entity_end_idx) - 1
                    cur_entity_str = " ".join(words[cur_entity_start_word_idx:cur_entity_end_word_idx+1])
                    entities.append(cur_entity_str)
                    cur_entity = [idx]
                    cur_ner_id = ner_id
            else:
                if not cur_entity:
                    cur_entity = [idx]
                    cur_ner_id = ner_id + 3
                else:
                    cur_entity_end_idx = cur_entity[-1]
                    if cur_entity_end_idx == idx - 1:
                        cur_entity.append(idx)
                    else:
                        cur_entity_start_idx = cur_entity[0]
                        cur_entity_end_idx = cur_entity[-1]
                        cur_entity_start_word_idx = bisect.bisect_right(word_idxs, cur_entity_start_idx) - 1
                        cur_entity_end_word_idx = bisect.bisect_right(word_idxs, cur_entity_end_idx) - 1
                        cur_entity_str = " ".join(words[cur_entity_start_word_idx:cur_entity_end_word_idx+1])
                        entities.append(cur_entity_str)
                        cur_entity = [idx]
                        cur_ner_id = ner_id + 3
            idx += 1
        if cur_entity:
            cur_entity_start_idx = cur_entity[0]
            cur_entity_end_idx = cur_entity[-1]
            cur_entity_start_word_idx = bisect.bisect_right(word_idxs, cur_entity_start_idx) - 1
            cur_entity_end_word_idx = bisect.bisect_right(word_idxs, cur_entity_end_idx) - 1
            cur_entity_str = " ".join(words[cur_entity_start_word_idx:cur_entity_end_word_idx+1])
            entities.append(cur_entity_str)

        # print(entities)
        # assert False
        
        # 去重、去空、去过短，去special token
        filtered_entities = []
        for e in entities:
            e = e.replace("[PAD]","").replace("[CLS]","").replace("[SEP]","").replace("[UNK]","").strip()
            if len(e) < 3:
                continue
            if e not in filtered_entities:
                filtered_entities.append(e)
        entities = filtered_entities
        # print(entities)

        restored_entities = []
        # 将entity还原为原本的大小写格式
        for entity_str in entities:
            try:
                entity_str_post = entity_str
                entity_str_post = entity_str_post.replace("( ","(").replace(" )",")")
                entity_str_post = entity_str_post.replace(" .",".").replace(" ,",",").replace(r' "', r'"')
                entity_str_post = entity_str_post.replace(" !","!").replace(" '","'").replace("' ","'").replace(" -","-").replace("- ","-")
                entity_str_post = entity_str_post.replace(" ?","?").replace(" / ",'/').replace("= ","=").replace("< ","<").replace("> ",">").replace(" : ",":")
                entity_start_idx = prompt.lower().index(entity_str_post)
                restored_entities.append(prompt[entity_start_idx:entity_start_idx+len(entity_str_post)])
            except Exception as e:
                print(f"entity_str_post:{entity_str_post}")
                print(f"prompt.lower():{prompt.lower()}")
                # assert False
                global error_count
                error_count += 1
        entities = restored_entities

        batch_entities.append(entities)
    return batch_entities

bsz = 16

all_ner_results = []

for batch_ins in tqdm(batch_list_generator(instructions, bsz),total=len(instructions)//bsz):
    input_batch_entities = batch_ner([ins['input'].strip() for ins in batch_ins], ner_model, ner_tok)
    output_batch_entities = batch_ner([ins['output'].strip() for ins in batch_ins], ner_model, ner_tok)
    batch_results = [{'input':batch_ins[i]['input'].strip(), 'input_entities':input_batch_entities[i],'output':batch_ins[i]['output'].strip(), 'output_entities':output_batch_entities[i],} for i in range(len(input_batch_entities))]
    all_ner_results.extend(batch_results)
print(f"error_entity_count:{error_count}")
json.dump(all_ner_results, open("data/ner_results.json", "w"))