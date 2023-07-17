import re
import bisect
import pylcs
import pandas as pd
import json
import jsonlines

df = pd.read_csv("data/umls_kg_filter_count_5_with_def.csv")
slice_start = 44000
size = 10000
ner_results = json.load(open("data/ner_results_chat_history_all.json", "r"))
ner_results = ner_results

from collections import defaultdict
s2t = defaultdict(list)
t2s = defaultdict(list)
edge_white_list = ['has active ingredient',
                   'has causative agent',
                   'has pathological process',
                   'possibly equivalent to',
                   'has definition of',
                   ]

for row in df.itertuples():
    tri_id = row[0]
    source = str(row.source)
    target = str(row.target)
    edge = row.edge
    if edge not in edge_white_list:
        continue
    s2t[source].append(tri_id)
    t2s[target].append(tri_id)

print("s2t", len(s2t.keys()), "t2s", len(t2s.keys()))

class Searcher:
    def __init__(self, keys):
        self.keys = keys
        self.his = {}
    
    def judge(self, str1: str, str2: str) -> bool:
        words1 = str1.lower().split()
        words2 = str2.lower().split()

        start_index = 0
        word1 = words1[0]

        for i in range(len(words2)):
            if words2[i] == word1:
                start_index = i
                for j in range(1, len(words1)):
                    if i + j < len(words2) and words1[j] == words2[i + j]:
                        continue
                    else:
                        break
                else:
                    return True

        return False

    def in_bf(self, q):
        if self.his.get(q):
            return self.his[q]
        else:
            res = [k for k in self.keys if self.judge(q, k)]
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

def get_entities_triplets(entities, threshold=50):
    all_entities = []
    all_triplets = []
    for e in entities:
        triplets = []
        
        source_keys = s2t_searcher.in_bf(e)
        
        # target_keys = t2s_searcher.in_bf(e)
        # if len(source_keys) + len(target_keys) > threshold or len(source_keys) + len(target_keys) == 0:
        #     continue
        # for k in target_keys:
        #     triplets.extend(t2s[k])
        if len(source_keys) > threshold or len(source_keys) == 0:
            continue
        
        for k in source_keys:
            triplets.extend(s2t[k])
        
        triplets = list(set(triplets))
        all_entities.append(e)
        all_triplets.append(triplets)
    return all_entities, all_triplets


from tqdm.auto import tqdm

from copy import deepcopy
grounding_results = deepcopy(ner_results)

def get_et(res):
    in_e, in_t = get_entities_triplets(res['input_entities'])
    out_e, out_t = get_entities_triplets(res['output_entities'])
    return {
        "input_entities": in_e,
        "input_triplets": in_t,
        "output_entities": out_e,
        "output_triplets": out_t,
    }

# for i, res in enumerate(tqdm(ner_results,total=len(ner_results))):
#     grounding_results[i].update(get_et(res))

from multiprocessing import Pool, cpu_count

process_num = cpu_count()
pool = Pool(process_num)
print(f"process_num: {process_num}")
for i, output in enumerate(tqdm(pool.imap(get_et, ner_results), total=len(ner_results))):
    grounding_results[i].update(output)

json.dump(grounding_results, open(f"data/kg_chat_{len(ner_results)}.json", "w"))