import os, json
from tqdm import tqdm
import pdb

base_dir = '/home/work/data/hangyul/mimic_cxr_not_filtered_refined'
original_dir = os.path.join(base_dir, 'medgemma_merged.json')
save_dir = os.path.join(base_dir, 'mimic_cxr_merged.json')
output_dict = {'train':{}, 'val':{}, 'test':{}}

with open(original_dir, 'r') as f: original_json = json.load(f)
for k, v in tqdm(original_json.items()):
    if v['in_mscxr'] is False: output_dict[v['split']][k] = v

with open(save_dir, 'w') as f: 
    json.dump(output_dict, f)