from transformers import AutoModelForCausalLM, AutoTokenizer
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
from multiprocessing import Pool, cpu_count

def prepare_mt():
    global dash_token_id
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    embedding = model.model.embed_tokens.weight
    old_shape = embedding.shape
    tok.add_tokens([dash_token])
    model.resize_token_embeddings(len(tok))
    dash_token_id = tok.convert_tokens_to_ids(dash_token)
    # Add the new token to the end of the embedding table
    print(f"DASH token is added to tokenizer and model\nDASH token id:{dash_token_id}\nEmbedding shape change from:{old_shape} to {embedding.shape}")


def pad_inputs(batch, pad_token_id=None):
    '''(input_ids:list[tensor], attention_mask:list[tensor, shape=seq*seq], labels:list[tensor])'''
    input_ids, attention_mask = batch[0], batch[1]
    labels = batch[2] if len(batch) == 3 else None

    max_len = max([x.shape[-1] for x in input_ids])
    input_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=pad_token_id) for x in input_ids]).squeeze()
    attention_mask = torch.stack([F.pad(x, (max_len - x.shape[-1], 0, 0, max_len - x.shape[-1]), mode='constant', value=0) for x in attention_mask]).squeeze()
    labels = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-100) for x in labels]).squeeze() if labels else None

    return input_ids, attention_mask, labels


class EntityDataset(Dataset):
    def __init__(self, data_path: str, kg_path : str, tokenizer: Tokenizer, size=None, max_len=1024, from_pickle=None, *args, **kwargs):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.kg = pd.read_csv(kg_path)
        self.data = json.load(open(data_path))
        self.data = self.data[:size] if size else self.data
        if from_pickle:
            self.input_ids, self.attention_mask, self.labels, self.prompts = pickle.load(open(from_pickle, "rb"))
            print(f"Loaded dataset from pickle file {from_pickle}")
        else:
            self.input_ids, self.attention_mask, self.labels, self.prompts = self.make_inputs()
            pickle.dump((self.input_ids, self.attention_mask, self.labels, self.prompts), open(f"EntityDataset_{len(self)}.pkl", "wb"))
            print(f"dump dataset to pickle file EntityDataset_{len(self)}.pkl")
        print(f"Loaded dataset with {len(self)} elements")
    

    def make_inputs(self):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        prompt_list = []

        # process_num = 1
        # pool = Pool(process_num)
        # print("Pool created with {process_num} process")

        make_input_func = partial(self.make_input_func, kg=self.kg, tok=self.tokenizer)
        # for output in tqdm(pool.imap(make_input_func, self.data),total=len(self.data)):
        for ins in tqdm(self.data, total=len(self.data)):
            output = make_input_func(ins)
            input_ids_list.append(output[0])
            attention_mask_list.append(output[1])
            labels_list.append(output[2])
            prompt_list.append(output[3])
        
        # pool.close()
        
        return input_ids_list, attention_mask_list, labels_list, prompt_list
    
    def make_input_func(self, ins, kg, tok):
        PROMPT_TEMPLATE = 'You are a doctor. Given the following patient information, write answer.\n\n##Input:\n{input}\n\n##Output:\n{output}'
        all_text = PROMPT_TEMPLATE.format(input=ins['input'], output=ins['output'])
        # print(f"all_text:{all_text}")
        inp = tok(all_text)
        input_ids, attention_mask = inp['input_ids'], inp['attention_mask']
        et = list({e:random.choice(t) for e,t in zip(ins['input_entities']+ins['output_entities'], ins['input_triplets']+ins['output_triplets']) if len(t) > 0}.items())
        et = sorted(et, key=lambda et: - len(et[0])) # 按entity的长度从长到短排序
        # print(f"et:{et}")
        entities_idxs = [-1] * len(input_ids)
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

        # 拓展input_ids并计算每个token的hard_position_type_id, 0代表non-entity tokens，1代表entity tokens, 2代表triplet tokens
        hard_position_type_ids = []
        idx = 0
        new_input_ids = []
        while idx < len(input_ids):
            if input_ids[idx] <= 0:
                e, t = et[-input_ids[idx]]
                # tid = random.choice(t)
                # et[i][1].remove(tid)
                tid = t
                triplet = kg.loc[tid]
                # print(f"e:{e}\ntriplet source: {triplet.source}\ntriplet edge: {triplet.edge}\ntriplet target: {triplet.target}")
                if e.lower() in triplet.source.lower():
                    e_idx = triplet.source.lower().index(e.lower())
                    source = triplet.source[:e_idx] + e + triplet.source[e_idx+len(e):]
                    tri_prompt = f"{source} {dash_token} {triplet.edge}"
                    tri_target = triplet.target
                elif e.lower() in triplet.target.lower():
                    e_idx = triplet.target.lower().index(e.lower())
                    t = triplet.target[:e_idx] + e + triplet.target[e_idx+len(e):]
                    tri_prompt = f"{t} {dash_token} {triplet.edge}"
                    tri_target = triplet.source
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
        # print(f"new_input_ids:{new_input_ids} length:{len(new_input_ids)}")
        # print(f"new_input:{tok.decode(new_input_ids)}")
        # print(f"hard_position_type_ids:{hard_position_type_ids} length:{len(hard_position_type_ids)}")

        # 利用hard_position_type_ids计算attention_mask_map, shape为seq_len*seq_len
        # type0可以看见type0和type1, type1可以看见type0,type1, type2可以看见type1,type2和type3, type3可以看见type1,type2和type3
        attention_mask = torch.tril(torch.ones(seq_len,seq_len))
        for i in range(seq_len):
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
        
        # print(f"attention_mask:{attention_mask}")
        blank_prompt = PROMPT_TEMPLATE.format(input="",output="")
        begin_out_ids = tok(blank_prompt)['input_ids'][-5:]
        begin_out_idxs = [i for i in range(len(input_ids)) if input_ids[i:i+len(begin_out_ids)] == begin_out_ids]
        if begin_out_idxs == []:
            print(f"output_start_ids:{begin_out_ids}\n{tok.batch_decode(begin_out_ids)}")
            print(f"input_ids:{input_ids}\n{tok.batch_decode(input_ids)}")
            assert begin_out_idxs != []
        output_start_idx = begin_out_idxs[0] + len(begin_out_ids)
        
        input_ids = torch.tensor(input_ids)
        labels = input_ids.clone()
        labels[:output_start_idx] = -100
        for i in range(len(hard_position_type_ids)):
            if hard_position_type_ids[i] == 3:
                labels[i] = input_ids[i]
        prompt = tok.decode(input_ids)

        return input_ids, attention_mask, labels, prompt
    def collate_fn(self, batch):
        input_ids = [b[0] for b in batch]
        attention_mask = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        return pad_inputs((input_ids, attention_mask, labels), self.tokenizer.pad_token_id)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

def train():
    for epoch in range(epoch_num):
        print(f"Epoch: {epoch}")
        for input_ids, attention_mask, labels in dl:
            print(f"input_ids:{input_ids}\n{tok.batch_decode(input_ids)}")
            print(f"attention_mask:{attention_mask}")
            print(f"labels:{labels}")
            input_ids.to(device)
            attention_mask.to(device)
            labels.to(device)
            res = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            print(res)
            break


# random.seed(42)
from_pickle=None
from_pickle="/mnt/workspace/guoyiqiu/coding/kg_llm/data/EntityDataset_1000.pkl"

mt_path = '/mnt/workspace/guoyiqiu/coding/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8'
mt_path = '/mnt/workspace/guoyiqiu/coding/huggingface/my_models/book-fast-tokenizer'
mt_path = '/mnt/workspace/guoyiqiu/coding/huggingface/my_models/Book_7B/checkpoint-4968'

kg_path = "data/umls_kg_filter.csv"
data_path = "data/kg_instruction_1000.json"
batch_size = 2
epoch_num = 1
dash_token = "[DASH]"
device = "cpu"

model = AutoModelForCausalLM.from_pretrained(mt_path).to(device)
tok = AutoTokenizer.from_pretrained(mt_path)
kg_instruction = json.load(open(data_path))
kg = pd.read_csv(kg_path)
prepare_mt()
dataset = EntityDataset(data_path, kg_path, tok, from_pickle=from_pickle)
dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
train()