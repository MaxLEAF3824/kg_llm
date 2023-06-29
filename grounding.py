import re
import bisect
import pylcs
import pandas as pd
import json
import jsonlines

df = pd.read_csv("data/umls_kg_filter.csv")
ner_results = json.load(open("data/ner_results.json", "r"))
instuctions = list(jsonlines.open('data/instruction_dataall.jsonl'))[:1000]
s2t = {}
t2s = {}
edge_white_list = ['has active ingredient',
                   'has causative agent',
                   'has direct procedure site',
                   'has dose form',
                   'has occurrence',
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
    s2t_leaf = {target:tri_id}
    t2s_leaf = {source:tri_id}
    if s2t.get(source):
        if s2t[source].get(edge):
            s2t[source][edge].append(s2t_leaf)
        else:
            s2t[source][edge] = [s2t_leaf]
    else:
        s2t[source] = {edge: [s2t_leaf]}
    if t2s.get(target):
        if t2s[target].get(edge):
            t2s[target][edge].append(t2s_leaf)
        else:
            t2s[target][edge] = [t2s_leaf]
    else:
        t2s[target] = {edge: [t2s_leaf]}

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
        threshold = max([len(w) for w in q.split(" ")])
        if self.his.get(q):
            return self.his[q]
        else:
            res = [k for k in self.keys if pylcs.lcs_sequence_length(q, k) >= threshold]
            self.his[q] = res
            return res

s2t_searcher = Searcher(s2t.keys())
t2s_searcher = Searcher(t2s.keys())

for i,res in enumerate(ner_results):
    in_entities = res['input']
    out_entities = res['output']
    in_source_keys = [s2t_searcher.lcs_bf(e) for e in in_entities]
    in_target_keys = [t2s_searcher.lcs_bf(e) for e in in_entities]
    out_source_keys = [s2t_searcher.lcs_bf(e) for e in out_entities]
    out_target_keys = [t2s_searcher.lcs_bf(e) for e in out_entities]
    in_related = [{"related_sources":in_source_keys, "related_targets":in_target_keys}]
    out_related = [{"related_sources":out_source_keys, "related_targets":out_target_keys}]
    instuctions[i].update({
        "input_grounding": {"entities": in_entities, 
                            "related_sources":in_source_keys, 
                            "related_targets":in_target_keys},
        "output_grounding": {"entities": out_entities, 
                            "related_sources":out_source_keys, 
                            "related_targets":out_target_keys},
    })

json.dump(instuctions, open("data/instruction_grounding_1000.json", "w"), indent=4)