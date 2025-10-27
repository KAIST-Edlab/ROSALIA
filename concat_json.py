import os, json
import pdb

base_dir = '/home/work/data/hangyul/mimic_cxr_not_filtered_qa'
train_dir = os.path.join(base_dir, 'medgemma_train.json')
val_dir = os.path.join(base_dir, 'medgemma_val.json')
test_dir = os.path.join(base_dir, 'medgemma_test.json')
save_dir = os.path.join(base_dir, 'medgemma_merged_final.json')
output_dict = {}

with open(train_dir, 'r') as f: train_json = json.load(f)
with open(val_dir, 'r') as f: val_json = json.load(f)
with open(test_dir, 'r') as f: test_json = json.load(f)

output_dict['train'] = train_json
output_dict['val'] = val_json
output_dict['test'] = test_json

with open(save_dir, 'w') as f: 
    json.dump(output_dict, f)