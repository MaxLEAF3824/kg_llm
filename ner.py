from transformers import AutoTokenizer, AutoModelForTokenClassification
import jsonlines
import pandas as pd
import random
import torch
from tqdm.auto import tqdm
import ujson as json

instructions = list(jsonlines.open('data/instruction_dataall.jsonl'))
df = pd.read_csv("data/umls_kg_filter.csv")

mt_path = "/mnt/workspace/guoyiqiu/coding/huggingface/my_models/RohanVB_umlsbert_ner"

ner_model = AutoModelForTokenClassification.from_pretrained(mt_path)
ner_tok = AutoTokenizer.from_pretrained(mt_path)
ner_model = ner_model.cuda()
ner_model.eval()


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
    for input_ids, ner_output in zip(inp['input_ids'], batch_ner_output):
        # word_idxs = [idx for idx, i in enumerate(input_ids) if not ner_tok.convert_ids_to_tokens(i).startswith("##")]
        nonzero_idxs = torch.nonzero(ner_output).squeeze().cpu().numpy().tolist()
        nonzero_idxs = [nonzero_idxs] if isinstance(nonzero_idxs, int) else nonzero_idxs
        
        if len(nonzero_idxs) == 0: # no ner words
            batch_entities.append([])
            continue
        
        last_idx = nonzero_idxs[0]
        entities = []
        cur_entity = []
        
        for idx in nonzero_idxs:
            ner_id = ner_output[idx].item()
            # B TAG: 开始新实体
            if ner_id >= 4: 
                # 结束当前实体并开始新实体
                if cur_entity:
                    entities.append(ner_tok.decode(cur_entity, skip_special_tokens=True))
                    cur_entity.clear()
                cur_entity.append(input_ids[idx])
            
            # I TAG 且idx连续: 继续当前实体 （不考虑是什么I TAG）   
            elif idx == last_idx + 1: 
                cur_entity.append(input_ids[idx])
            # I TAG 且idx不连续: 开始新实体
            else:
                # 结束当前实体并开始新实体
                if cur_entity:
                    entities.append(ner_tok.decode(cur_entity, skip_special_tokens=True))
                    cur_entity.clear()
                cur_entity.append(input_ids[idx])
            last_idx = idx
        
        if cur_entity:
            entities.append(ner_tok.decode(cur_entity, skip_special_tokens=True))
            cur_entity.clear()

        entities = [e for e in set(entities) if e and len(e)>=3]

        batch_entities.append(entities)
    return batch_entities

bsz = 16

all_ner_results = []

for batch_ins in tqdm(batch_list_generator(instructions[:1000], bsz),total=len(instructions)//bsz):
    input_batch_entities = batch_ner([ins['input'].strip() for ins in batch_ins], ner_model, ner_tok)
    output_batch_entities = batch_ner([ins['output'].strip() for ins in batch_ins], ner_model, ner_tok)
    batch_results = [{'input_entities':input_batch_entities[i], 'output_entities':output_batch_entities[i],} for i in range(len(input_batch_entities))]
    all_ner_results.extend(batch_results)

json.dump(all_ner_results, open("data/ner_results.json", "w"))