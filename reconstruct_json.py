import os, json
from tqdm import tqdm
import pdb

base_dir = '/home/work/data/hangyul/mimic_cxr_final'
original_dir = os.path.join(base_dir, 'medgemma_merged_1105_2.json')
final_dir = '/home/work/data/hangyul/mimic_cxr_final/test.json'
save_dir = os.path.join(base_dir, 'mimic_cxr_merged_final.json')
output_dict = {'train':{}, 'val':{}, 'test':{}}

with open(original_dir, 'r') as f: original_json = json.load(f)
with open(final_dir, 'r') as f: final_test_json = json.load(f)
for k, v in tqdm(original_json.items()):
    if v['in_mscxr'] is False: 
        if v['split'] != 'test': output_dict[v['split']][k] = v
        else: 
            if k not in final_test_json.keys(): del output_dict[v['split']][k]
            output_dict[v['split']][k] = final_test_json[k]

with open(save_dir, 'w') as f: 
    json.dump(output_dict, f)