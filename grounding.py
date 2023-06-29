import re
import bisect
import pylcs
import pandas as pd
import json
import jsonlines

df = pd.read_csv("data/umls_kg_filter.csv")
size = 1000
ner_results = json.load(open("data/ner_results.json", "r"))[:size]
instuctions = list(jsonlines.open('data/instruction_dataall.jsonl'))[:size]

from collections import defaultdict
s2t = defaultdict(list)
t2s = defaultdict(list)
edge_white_list = ['has active ingredient',
                   'has causative agent',
                #    'has direct procedure site',
                #    'has dose form',
                #    'has occurrence',
                   'has pathological process',
                   'possibly equivalent to'
                   ]

for row in df.itertuples():
    tri_id = row[0]
    source = row.source.lower()
    target = row.target.lower()
    edge = row.edge.lower()
    if edge not in edge_white_list:
        continue
    s2t[source].append(tri_id)
    t2s[target].append(tri_id)

print("s2t", len(s2t.keys()), "t2s", len(t2s.keys()))

class Searcher:
    def __init__(self, keys):
        self.keys = keys
        self.keys_str, self.keys_idx = self.build(keys)
        self.his = {}
    
    def build(self, keys):
        keys_str = ''.join(keys)
        len_sum = 0
        keys_idx = []
        for k in keys:
            keys_idx.append(len_sum)
            len_sum += len(k)
        return keys_str, keys_idx

    def in_re(self, q):
        if self.his.get(q):
            return self.his[q]
        else:
            pattern = re.compile(re.escape(q))
            matches = []
            for m in pattern.finditer(self.keys_str):
                start_pos_idx = bisect.bisect_right(self.keys_idx, m.start()) - 1
                start_pos = self.keys_idx[start_pos_idx]
                end_pos = self.keys_idx[start_pos_idx + 1] if start_pos_idx + 1 < len(self.keys_idx) - 1 else -1
                matches.append(self.keys_str[start_pos:end_pos])
            self.his[q] = matches
        return matches

    def in_bf(self, q):
        if self.his.get(q):
            return self.his[q]
        else:
            res = [k for k in self.keys if q in k]
            self.his[q] = res
            return res
    
    def lcs_bf(self, q:str):
        threshold = max(int(0.8*len(q)),3)
        # threshold = max([len(w) for w in q.split()])
        if self.his.get(q):
            return self.his[q]
        else:
            res = [k for k in self.keys if pylcs.lcs_string_length(q, k) >= threshold]
            self.his[q] = res
            return res

s2t_searcher = Searcher(s2t.keys())
t2s_searcher = Searcher(t2s.keys())
from tqdm.auto import tqdm
for i,res in tqdm(enumerate(ner_results), total=len(ner_results)):
    in_entities = []
    out_entities = []
    
    in_source_keys = []
    in_target_keys = []
    out_source_keys = []
    out_target_keys = []
    for e in res['input_entities']:
        in_source_key = s2t_searcher.in_bf(e)
        in_target_key = t2s_searcher.in_bf(e)
        if len(in_source_key)+len(in_target_key) > 500:
            continue
        if len(in_source_key) == 0 and len(in_target_key) == 0:
            continue
        in_source_keys.append(in_source_key)
        in_target_keys.append(in_target_key)
        in_entities.append(e)
        
    for e in res['output_entities']:
        out_source_key = s2t_searcher.in_bf(e)
        out_target_key = t2s_searcher.in_bf(e)
        if len(out_source_key)+len(out_target_key) > 500:
            continue
        if len(out_source_key) == 0 and len(out_target_key) == 0:
            continue
        out_source_keys.append(out_source_key)
        out_target_keys.append(out_target_key)
        out_entities.append(e)
        
    instuctions[i].update({
        "input_grounding": {"entities": in_entities, 
                            "related_sources":in_source_keys, 
                            "related_targets":in_target_keys},
        "output_grounding": {"entities": out_entities, 
                            "related_sources":out_source_keys, 
                            "related_targets":out_target_keys},
    })

json.dump(instuctions, open("data/instruction_grounding_1000.json", "w"), )